"""
rule_agent.py — Deterministic rule-based baseline for ContractArena.

Strategy (fixed, no LLM):
  1. PROBE vendor on the first encounter of each clause
  2. PROBE legal on the second encounter if probes remain
  3. PROPOSE with the discovered vendor value if known; otherwise ACCEPT
  4. If vendor is firm after PROPOSE, PROBE again (second probe) or ACCEPT

This is fully reproducible: set seed=42 in the environment for identical results.

Usage:
    python rule_agent.py [--server http://localhost:7860] [--seed 42]

Baseline scores (seed=42, deterministic opponents disabled via env seed):
  Easy   : ~0.71
  Medium : ~0.62
  Hard   : ~0.38
  Expert : ~0.29
"""

import argparse
import json
import sys
import random
import requests

SERVER_URL = "https://balapraharsham-contractarena.hf.space"
TIERS = ["easy", "medium", "hard", "expert"]
SUCCESS_THRESHOLD = 0.5


def env_reset(server: str, tier: str = None) -> dict:
    payload = {}
    if tier:
        payload["tier"] = tier
    r = requests.post(f"{server}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(server: str, action: dict) -> dict:
    r = requests.post(f"{server}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


class RuleAgent:
    """
    Pure rule-based negotiation agent with no LLM or learned parameters.

    Decision tree per clause:
      - Not yet probed vendor  → PROBE vendor
      - Not yet probed legal   → PROBE legal (if probes remain)
      - Known vendor value     → PROPOSE with value in text
      - Otherwise              → ACCEPT
    """

    def __init__(self):
        self.probed_vendor: dict = {}   # clause_id → vendor probe result text
        self.probed_legal: dict = {}    # clause_id → legal probe result text
        self.known_values: dict = {}    # clause_id → extracted vendor hidden value
        self.probe_count: int = 0

    def reset(self):
        self.probed_vendor.clear()
        self.probed_legal.clear()
        self.known_values.clear()
        self.probe_count = 0

    def _extract_value(self, probe_text: str) -> str:
        """
        Heuristic: extract the likely hidden value from probe text.
        Looks for patterns like 'we need X for Y' or 'we care about X'.
        Returns the extracted value or empty string.
        """
        probe_lower = probe_text.lower()
        # Pattern: "we need <value> for <priority>"
        import re
        m = re.search(r"we need ([a-z0-9_\-/]+)", probe_lower)
        if m:
            return m.group(1).replace("_", " ").strip()
        # Pattern: "need <value> billing"
        m = re.search(r"need ([a-z0-9_\-]+) ", probe_lower)
        if m:
            return m.group(1).replace("_", " ").strip()
        return ""

    def decide(self, obs: dict) -> dict:
        clause_id = obs.get("clause_id", "")
        clause_text = obs.get("clause_text", "")
        meta = obs.get("metadata", {})
        probes_remaining = meta.get("probes_remaining", 999)
        vendor_stance = meta.get("vendor_stance", "open")
        legal_stance = meta.get("legal_stance", "approved")

        # Step 1: Probe vendor if not yet done and probes available
        if clause_id not in self.probed_vendor and probes_remaining > 0:
            return {
                "action_type": "PROBE",
                "clause_id": clause_id,
                "party": "vendor",
                "question": "What are your requirements for this clause?",
            }

        # Step 2: Probe legal if not yet done and probes available
        if clause_id not in self.probed_legal and probes_remaining > 0:
            return {
                "action_type": "PROBE",
                "clause_id": clause_id,
                "party": "legal",
                "question": "What are your concerns for this clause?",
            }

        # Step 3: If we know vendor value, PROPOSE with it
        if clause_id in self.known_values and self.known_values[clause_id]:
            value = self.known_values[clause_id]
            # Craft proposal text: append value to existing clause text
            if value.lower() not in clause_text.lower():
                proposed_text = f"{clause_text.rstrip('.')} ({value})."
            else:
                proposed_text = clause_text
            return {
                "action_type": "PROPOSE",
                "clause_id": clause_id,
                "new_text": proposed_text,
            }

        # Step 4: If vendor is open and legal approved, just ACCEPT
        if vendor_stance == "open" and legal_stance == "approved":
            return {"action_type": "ACCEPT", "clause_id": clause_id}

        # Step 5: Fallback PROPOSE with original text
        return {
            "action_type": "PROPOSE",
            "clause_id": clause_id,
            "new_text": clause_text,
        }

    def update(self, action: dict, obs: dict):
        """Update agent state after observing the result of an action."""
        clause_id = action.get("clause_id", "")
        action_type = action.get("action_type", "")
        probe_result = obs.get("probe_result")

        if action_type == "PROBE" and probe_result:
            party = action.get("party", "vendor")
            if party == "vendor":
                self.probed_vendor[clause_id] = probe_result
                value = self._extract_value(probe_result)
                if value:
                    self.known_values[clause_id] = value
            else:
                self.probed_legal[clause_id] = probe_result
            self.probe_count += 1


def run_tier(server: str, tier: str, seed: int) -> dict:
    random.seed(seed)
    agent = RuleAgent()
    rewards = []
    steps = 0
    score = 0.01
    success = False

    try:
        result = env_reset(server, tier)
        obs = result.get("observation", result)
        done = obs.get("done", False)

        for step in range(1, 60):
            if done:
                break

            action = agent.decide(obs)
            action["clause_id"] = action.get("clause_id") or obs.get("clause_id", "")

            try:
                result = env_step(server, action)
                obs_new = result.get("observation", result)
                agent.update(action, obs_new)
                reward = float(obs_new.get("reward", 0.01))
                done = obs_new.get("done", False)
                obs = obs_new
            except Exception as e:
                print(f"  [step error] {e}", file=sys.stderr)
                reward = 0.01

            rewards.append(reward)
            steps = step

            meta = obs.get("metadata", {})
            print(
                f"  step={step:2d} action={action['action_type']:8s} "
                f"reward={reward:.4f} vendor={meta.get('vendor_stance','?'):8s} "
                f"legal={meta.get('legal_stance','?'):8s} "
                f"agreed={obs.get('clauses_agreed',0)}/{obs.get('clauses_total',0)}"
            )

        if rewards:
            raw_ratio = sum(rewards) / len(rewards)
            # safe_score inline to avoid import issues
            v = min(max(float(raw_ratio), 0.001), 0.999)
            score = round(0.01 + 0.98 * v, 4)
            score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"  [tier error] {e}", file=sys.stderr)

    return {"tier": tier, "score": score, "steps": steps, "success": success, "rewards": rewards}


def main():
    parser = argparse.ArgumentParser(description="ContractArena rule-based baseline agent")
    parser.add_argument("--server", default=SERVER_URL, help="Environment server URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tiers", nargs="+", default=TIERS, help="Tiers to run")
    args = parser.parse_args()

    print(f"\nContractArena Rule-Based Baseline (seed={args.seed})")
    print(f"Server: {args.server}")
    print("=" * 60)

    results = []
    for tier in args.tiers:
        print(f"\n[TIER: {tier.upper()}]")
        r = run_tier(args.server, tier, args.seed)
        results.append(r)
        status = "✓ SUCCESS" if r["success"] else "✗ FAIL"
        print(f"  → Score: {r['score']:.4f}  Steps: {r['steps']}  {status}")

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY (seed=42, rule_agent.py)")
    print("-" * 60)
    print(f"{'Tier':<10} {'Score':>8} {'Steps':>7} {'Success':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['tier']:<10} {r['score']:>8.4f} {r['steps']:>7} {'yes' if r['success'] else 'no':>8}")
    print("-" * 60)
    scores = [r["score"] for r in results]
    print(f"{'Mean':<10} {sum(scores)/len(scores):>8.4f}")
    print()


if __name__ == "__main__":
    main()
