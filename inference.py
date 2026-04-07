import os
import sys
from openai import OpenAI
import requests

# ── config ───────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")
MAX_STEPS    = 15
SUCCESS_THRESHOLD = 0.5

TIERS = ["easy", "medium", "hard"]

# ── logging — stdout only for required lines ──────────────────────────────
def log_start(task, model):
    print(f"[START] task={task} env=contractarena model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}", flush=True)

def log_end(success, steps, rewards):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

def debug(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

# ── env helpers ───────────────────────────────────────────────────────────
def env_reset():
    r = requests.post(f"{SERVER_URL}/reset", json={})
    r.raise_for_status()
    return r.json()

def env_step(action: dict):
    r = requests.post(f"{SERVER_URL}/step", json={"action": action})
    r.raise_for_status()
    return r.json()

# ── agent ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a business deal negotiator.
You negotiate contracts clause by clause against a Vendor and a Legal Reviewer.
Each has hidden requirements you must uncover.

Your available actions:
- PROBE: Ask a party what they want (use this first to uncover hidden agendas)
- ACCEPT: Accept the current clause as-is
- PROPOSE: Suggest modified clause text
- REJECT: Reject a clause with a reason
- ESCALATE: Force both parties to reveal their positions

Strategy:
1. Always PROBE vendor first, then PROBE legal to uncover hidden requirements
2. Then PROPOSE text that satisfies both hidden requirements
3. ACCEPT only if both parties seem cooperative

Respond ONLY with valid JSON matching this schema:
{
  "action_type": "PROBE" | "ACCEPT" | "REJECT" | "PROPOSE" | "ESCALATE",
  "clause_id": "<clause id string>",
  "party": "vendor" or "legal" (only for PROBE),
  "question": "<your question>" (only for PROBE),
  "new_text": "<proposed clause text>" (only for PROPOSE),
  "reason": "<reason>" (only for REJECT)
}"""

def get_action(client, obs: dict, history: list) -> dict:
    import json
    user_msg = f"""Current clause: {obs['clause_id']}
Clause text: {obs['clause_text']}
Vendor response: {obs['vendor_response']}
Legal response: {obs['legal_response']}
Probe result: {obs.get('probe_result')}
Clauses agreed: {obs['clauses_agreed']} / {obs['clauses_total']}
Rounds remaining: {obs['rounds_remaining']}
Recent history: {history[-4:]}

What is your next action? Respond with JSON only."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        # strip markdown fences if model adds them
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        return json.loads(raw)
    except Exception as e:
        debug(f"model error: {e}")
        return {
            "action_type": "PROBE",
            "clause_id": obs["clause_id"],
            "party": "vendor",
            "question": "What matters most to you?"
        }


def run_tier(client, tier: str):
    log_start(task=tier, model=MODEL_NAME)
    rewards = []
    history = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        result = env_reset()
        obs = result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            action = get_action(client, obs, history)
            action["clause_id"] = action.get("clause_id") or obs["clause_id"]

            try:
                result = env_step(action)
                obs = result["observation"]
                reward = float(result.get("reward") or 0.0)
                done = result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action["action_type"],
                     reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {action['action_type']} "
                f"clause={action.get('clause_id')} reward={reward:.2f}"
            )

            if done:
                break

        max_possible = 3.0
        score = sum(rewards) / max_possible
        score = round(min(max(score, 0.0), 1.0), 2)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        debug(f"tier error: {e}")

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


def main():
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    debug(f"Running ContractArena inference on {len(TIERS)} tiers")
    debug(f"Server: {SERVER_URL} | Model: {MODEL_NAME}")

    for tier in TIERS:
        debug(f"Starting tier: {tier}")
        score = run_tier(client, tier)
        debug(f"Tier {tier} final score: {score}")


if __name__ == "__main__":
    main()