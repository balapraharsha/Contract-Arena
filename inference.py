import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.5
TIERS = ["easy", "medium", "hard"]


def log_start(task, model):
    print(f"[START] task={task} env=contractarena model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success, steps, rewards):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


def debug(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def env_reset():
    r = requests.post(f"{SERVER_URL}/reset", json={})
    r.raise_for_status()
    return r.json()


def env_step(action: dict):
    r = requests.post(f"{SERVER_URL}/step", json={"action": action})
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = """You are a business deal negotiator.
You negotiate contracts clause by clause against a Vendor and a Legal Reviewer.
Each has hidden requirements you must uncover.

Available actions:
- PROBE
- ACCEPT
- PROPOSE
- REJECT
- ESCALATE

Respond ONLY in valid JSON.
"""


def get_action(client, obs: dict, history: list) -> dict:
    user_msg = f"""Current clause: {obs['clause_id']}
Clause text: {obs['clause_text']}
Vendor response: {obs['vendor_response']}
Legal response: {obs['legal_response']}
Probe result: {obs.get('probe_result')}
Clauses agreed: {obs['clauses_agreed']} / {obs['clauses_total']}
Rounds remaining: {obs['rounds_remaining']}
Recent history: {history[-4:]}

Return JSON only."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()

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
            "question": "What matters most to you?",
        }


def run_tier(client, tier: str):
    log_start(task=tier, model=MODEL_NAME)
    rewards = []
    history = []
    steps_taken = 0
    success = False

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
            log_step(step, action["action_type"], reward, done, error)

            history.append(
                f"Step {step}: {action['action_type']} clause={action.get('clause_id')} reward={reward:.2f}"
            )

            if done:
                break

        score = sum(rewards) / 3.0
        score = round(min(max(score, 0.0), 1.0), 2)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        debug(f"tier error: {e}")

    finally:
        log_end(success, steps_taken, rewards)

    return success


def main():
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for tier in TIERS:
        debug(f"Starting tier: {tier}")
        run_tier(client, tier)


if __name__ == "__main__":
    main()
 