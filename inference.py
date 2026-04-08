import os
import sys
import json
import requests
from openai import OpenAI

SERVER_URL = os.environ.get("SERVER_URL", "https://balapraharsham-contractarena.hf.space")
MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.5
TIERS = ["easy", "medium", "hard"]


def safe_score(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return round(0.01 + 0.98 * value, 4)


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

Available actions: PROBE, ACCEPT, PROPOSE, REJECT, ESCALATE

Respond ONLY in valid JSON.
Example: {"action_type": "PROBE", "clause_id": "pricing", "party": "vendor", "question": "What matters most to you?"}
"""


def get_action(client, model, obs: dict, history: list) -> dict:
    user_msg = f"""Current clause: {obs['clause_id']}
Clause text: {obs['clause_text']}
Vendor response: {obs['vendor_response']}
Legal response: {obs['legal_response']}
Probe result: {obs.get('probe_result')}
Clauses agreed: {obs['clauses_agreed']} / {obs['clauses_total']}
Rounds remaining: {obs['rounds_remaining']}
Recent history: {history[-4:]}

Return JSON only."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue

    return json.loads(raw)


def run_tier(client, model, tier: str):
    log_start(task=tier, model=model)
    rewards = []
    history = []
    steps_taken = 0
    success = False

    try:
        result = env_reset()
        obs = result["observation"]
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            try:
                action = get_action(client, model, obs, history)
            except Exception as e:
                debug(f"get_action error: {e}")
                action = {"action_type": "ACCEPT", "clause_id": obs["clause_id"]}

            action["clause_id"] = action.get("clause_id") or obs["clause_id"]

            if done:
                rewards.append(0.01)
                steps_taken = step
                log_step(step, action["action_type"], 0.01, True)
                break

            try:
                result = env_step(action)
                obs = result["observation"]
                reward = float(result.get("reward") or 0.01)
                reward = min(max(reward, 0.01), 0.99)
                done = result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.01
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action["action_type"], reward, done, error)
            history.append(f"Step {step}: {action['action_type']} clause={action.get('clause_id')} reward={reward:.2f}")

            if done:
                break

        total = sum(rewards)
        max_possible = len(rewards) if rewards else 1
        score = safe_score(total / max_possible)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        debug(f"tier error: {e}")

    finally:
        log_end(success, steps_taken, rewards)

    return success


def main():
    api_key = os.environ["API_KEY"]
    api_base = os.environ["API_BASE_URL"]
    model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    debug(f"API_BASE_URL={api_base}")
    debug(f"MODEL_NAME={model}")
    debug(f"SERVER_URL={SERVER_URL}")
    debug(f"API_KEY set: {bool(api_key)}")

    client = OpenAI(base_url=api_base, api_key=api_key)

    for tier in TIERS:
        debug(f"Starting tier: {tier}")
        run_tier(client, model, tier)


if __name__ == "__main__":
    main()