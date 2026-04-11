"""
train_grpo.py — GRPO fine-tuning pipeline for ContractArena.

Trains Qwen2.5-0.5B (or any HF causal LM) using Group Relative Policy
Optimisation on trajectories collected from the ContractArena environment.

Quickstart (no GPU needed for 0.5B):

    # 1. Start server
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # 2. Collect trajectories with rule agent
    python rule_agent.py --server http://localhost:8000
    python trajectory_collector.py export trajectories.jsonl grpo_data.jsonl

    # 3. Dry run — verify everything works
    python train_grpo.py --dry-run

    # 4. Train
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

    # 5. Push to HuggingFace
    python train_grpo.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --push-to-hub YOUR_USERNAME/contractarena-negotiator-qwen

Requirements:
    pip install trl transformers torch datasets peft accelerate
"""

import argparse
import json
import os
import re
import sys
import requests
from pathlib import Path
from typing import Any, Dict, List

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")


# ── GRPO reward function ──────────────────────────────────────────────────────

def contractarena_reward(
    completions: List[str],
    prompts: List[str],
    **kwargs,
) -> List[float]:
    """
    GRPO reward function.
    Parse each completion as a ContractArena action JSON,
    execute against the live environment, return the reward signal.
    """
    rewards = []
    for completion in completions:
        try:
            action = _parse_action(completion)
            if not action:
                rewards.append(0.01)
                continue
            r = requests.post(
                f"{SERVER_URL}/step",
                json={"action": action},
                timeout=10,
            )
            if r.status_code != 200:
                rewards.append(0.01)
                continue
            reward = float(r.json().get("reward", 0.01))
            rewards.append(min(max(reward, 0.01), 0.99))
        except Exception:
            rewards.append(0.01)
    return rewards


def _parse_action(text: str) -> Dict[str, Any]:
    """Extract a JSON action dict from model completion text."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    trajectory_file: str,
    max_examples: int = 1000,
) -> List[Dict]:
    """Convert JSONL trajectory file to GRPO training examples."""
    path = Path(trajectory_file)
    if not path.exists():
        print(f"No trajectory file at {trajectory_file} — using synthetic expert demos")
        return _synthetic_examples(max_examples)

    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except Exception:
                continue
            for step in ep.get("steps", []):
                obs    = step.get("observation", {})
                action = step.get("action", {})
                reward = step.get("reward", 0.01)
                examples.append({
                    "prompt":     _obs_to_prompt(obs, ep.get("tier", "easy")),
                    "completion": json.dumps(action),
                    "reward":     reward,
                    "tier":       ep.get("tier", "easy"),
                })
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break

    print(f"Built {len(examples)} training examples from {trajectory_file}")
    return examples


def _obs_to_prompt(obs: dict, tier: str) -> str:
    meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    feat = meta.get("numerical_features", {})
    feat_str = ""
    if feat:
        feat_str = (
            f"\n[pressure={feat.get('negotiation_pressure',0):.2f} "
            f"agreement={feat.get('clause_agreement_rate',0):.2f} "
            f"hostility={feat.get('vendor_hostility_index',0):.2f} "
            f"eff_ratio={meta.get('efficiency_ratio',0):.2f}]"
        )
    mk = meta.get("marathon_knowledge", {})
    mk_str = f"\nTransferred knowledge: {json.dumps(mk)}" if mk else ""

    return (
        f"Negotiate {tier}-tier contract.\n"
        f"Clause: {obs.get('clause_id','')} — {obs.get('clause_text','')}\n"
        f"Vendor ({meta.get('vendor_stance','?')}): {obs.get('vendor_response','')}\n"
        f"Legal ({meta.get('legal_stance','?')}): {obs.get('legal_response','')}\n"
        f"Probe result: {obs.get('probe_result') or 'none'}\n"
        f"Agreed: {obs.get('clauses_agreed',0)}/{obs.get('clauses_total',0)} "
        f"Rounds left: {obs.get('rounds_remaining',0)} "
        f"Probes left: {meta.get('probes_remaining','?')}"
        f"{feat_str}{mk_str}\n"
        f"Output a single JSON action. Example: "
        f'{"action_type": "PROBE", "clause_id": "pricing", "party": "vendor"}'
    )


def _synthetic_examples(n: int) -> List[Dict]:
    """Expert demonstrations for cold-start (no server required)."""
    clauses = [
        ("pricing",   "Subscription fee: $500/month, billed annually."),
        ("support",   "Vendor provides email support within 48 hours."),
        ("payment",   "Payment due within 60 days of delivery."),
        ("liability", "Vendor liability limited to contract value."),
    ]
    examples = []
    for i in range(n):
        cid, ctxt = clauses[i % len(clauses)]
        if i % 3 == 0:
            action = {"action_type": "PROBE", "clause_id": cid, "party": "vendor",
                      "question": "What are your key requirements for this clause?"}
            reward = 0.11
        elif i % 3 == 1:
            action = {"action_type": "PROPOSE", "clause_id": cid,
                      "new_text": f"{ctxt} with monthly billing, net_30, revenue_based."}
            reward = 0.41
        else:
            action = {"action_type": "ACCEPT", "clause_id": cid}
            reward = 0.41
        examples.append({
            "prompt": (
                f"Negotiate easy-tier contract.\n"
                f"Clause: {cid} — {ctxt}\n"
                f"Vendor (open): Ready to negotiate.\n"
                f"Legal (approved): Ready to review.\n"
                f"Output JSON action."
            ),
            "completion": json.dumps(action),
            "reward": reward,
            "tier": "easy",
        })
    return examples


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(args):
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import Dataset
        import torch
    except ImportError:
        print("Install training deps: pip install trl transformers torch datasets peft accelerate")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ContractArena GRPO Training")
    print(f"  Model:    {args.model}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  GPU:      {__import__('torch').cuda.is_available()}")
    print(f"{'='*60}\n")

    examples = build_dataset(args.trajectory_file, args.max_examples)

    if args.dry_run:
        print(f"✓ Dry run — {len(examples)} examples ready.")
        print(f"\nSample prompt:\n{examples[0]['prompt']}")
        print(f"\nSample completion: {examples[0]['completion']}")
        print(f"Sample reward:     {examples[0]['reward']}")
        print("\nAll good — remove --dry-run to train.")
        return

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    dataset = Dataset.from_list([{"prompt": e["prompt"]} for e in examples])

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_completion_length=250,
        max_prompt_length=700,
        num_generations=args.num_generations,
        logging_steps=5,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[contractarena_reward],
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✓ Model saved to {args.output_dir}")

    if args.push_to_hub:
        print(f"Pushing to HuggingFace: {args.push_to_hub} ...")
        trainer.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"✓ Published at https://huggingface.co/{args.push_to_hub}")
        print(f"\nAdd this badge to your README:")
        print(f'[![Model](https://img.shields.io/badge/🤗 Model-contractarena--negotiator-orange)](https://huggingface.co/{args.push_to_hub})')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ContractArena GRPO Training Pipeline")
    p.add_argument("--model",            default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID to fine-tune")
    p.add_argument("--trajectory-file",  default="trajectories.jsonl",
                   help="JSONL trajectories from trajectory_collector.py")
    p.add_argument("--output-dir",       default="training/grpo-contractarena",
                   help="Output directory for trained model")
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch-size",       type=int,   default=2)
    p.add_argument("--lr",               type=float, default=5e-6)
    p.add_argument("--num-generations",  type=int,   default=4,
                   help="GRPO group size — number of completions per prompt")
    p.add_argument("--max-examples",     type=int,   default=500,
                   help="Maximum training examples from trajectory file")
    p.add_argument("--push-to-hub",      default="",
                   help="HuggingFace repo to publish trained model (e.g. username/model-name)")
    p.add_argument("--dry-run",          action="store_true",
                   help="Build dataset and verify config without training")
    run_training(p.parse_args())
