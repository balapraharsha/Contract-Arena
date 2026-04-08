import os
import sys
import json
import requests
from openai import OpenAI

try:
    from server.utils import safe_score
    from trajectory_collector import TrajectoryCollector
    _COLLECTOR_AVAILABLE = True
except ImportError:
    _COLLECTOR_AVAILABLE = False
    def safe_score(value: float) -> float:
        value = min(max(float(value), 0.001), 0.999)
        result = round(0.01 + 0.98 * value, 4)
        return min(max(result, 0.01), 0.99)

SERVER_URL = os.environ.get("SERVER_URL", "https://balapraharsham-contractarena.hf.space")
MAX_STEPS = 30
SUCCESS_THRESHOLD = 0.5
TIERS = ["easy", "medium", "hard", "expert"]

# Initialise trajectory collector (logs to trajectories.jsonl)
_collector = TrajectoryCollector("trajectories.jsonl") if _COLLECTOR_AVAILABLE else None


def log_start(task, model):
    print(f"[START] task={task} env=contractarena model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done_str} error={err}", flush=True)


def log_end(success, steps, rewards, score):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


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


SYSTEM_PROMPT = """You are an expert contract negotiator. Your goal is to get ALL clauses agreed with maximum bonus.

=== SCORING RULES (know these to maximize your score) ===
- ACCEPT or PROPOSE when vendor=open AND legal=approved → +0.40 reward
- PROBE → +0.10 reward (reveals hidden requirements); redundant probes: -0.05
- Legal flags clause → -0.20 penalty (AVOID flagged patterns)
- Vendor walkout → -0.30 penalty (don't reject too many times)
- Vendor stance moves firm→open without full agreement → +0.05 shaping reward
- Proposed text fuzzy-matches vendor's hidden value → up to +0.08 shaping reward
- FINAL BONUS (only at episode end):
  * +0.20 if vendor's hidden_value appears in any agreed clause text
  * +0.15 if no legal redline patterns appear in any agreed text
  * +0.05 if rounds remaining > 0 when all clauses agreed
  * +0.10 (expert tier only) if compliance officer's required keyword appears in agreed text

=== EXPERT TIER EXTRA RULES ===
- A silent Compliance Officer also reviews clauses; probe with party="compliance" to learn their requirement
- Compliance blocks trigger -0.15 penalty
- Final score uses geometric mean of negotiation score × compliance score — one bad compliance hit tanks everything
- Probe compliance FIRST on data/audit clauses

=== OPTIMAL STRATEGY ===
1. At the START of a new clause: PROBE vendor first (party="vendor") to learn their hidden priority/value
2. After probing: craft a PROPOSE with new_text that includes the vendor's revealed value word
3. If vendor already open and legal approved: ACCEPT is fine
4. NEVER use ESCALATE (always causes legal to flag → -0.20)
5. NEVER REJECT more than twice for the same vendor (causes walkout at threshold — now random 2–4!)
6. Watch probes_remaining — on hard/expert tier you only have 3–4 total, use them on early clauses
7. If vendor stance is "firm": PROPOSE with their hidden value in the text
8. If legal stance is "flagged": rephrase to remove flagged terms, then PROPOSE again
9. Do NOT re-probe information already known — redundant probes now cost -0.05

=== HIDDEN VALUE EXPLOITATION ===
After a PROBE reveals the vendor wants e.g. "monthly" billing or "net_30" payment:
- Include that exact word/phrase in your PROPOSE new_text
- Example: new_text="Subscription fee: $500/month, billed monthly."
- This unlocks the +0.20 vendor bonus at episode end

=== ACTION FORMAT ===
Always respond with valid JSON only. Choose the best action:

PROBE (learn hidden requirements — do this first on each clause if probes available):
{"action_type": "PROBE", "clause_id": "<id>", "party": "vendor", "question": "What are your key requirements for this clause?"}

PROBE compliance (expert tier only, for data/audit clauses):
{"action_type": "PROBE", "clause_id": "<id>", "party": "compliance", "question": "What compliance requirements apply?"}

PROPOSE (when you know what vendor wants — include their value in new_text):
{"action_type": "PROPOSE", "clause_id": "<id>", "new_text": "<clause text that includes vendor's hidden value>"}

ACCEPT (when vendor=open and legal=approved and you don't need the vendor bonus):
{"action_type": "ACCEPT", "clause_id": "<id>"}

REJECT (use sparingly — max 1-2 times per clause or vendor walks out):
{"action_type": "REJECT", "clause_id": "<id>", "reason": "<reason>"}
"""


def get_action(client, model, obs: dict, history: list, probed_values: dict) -> dict:
    meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    vendor_stance = meta.get("vendor_stance", "unknown")
    legal_stance = meta.get("legal_stance", "unknown")
    probes_remaining = meta.get("probes_remaining", "unknown")
    agreed_clauses = meta.get("agreed_clauses", [])

    # Build context with all useful state
    user_msg = f"""=== CURRENT STATE ===
Clause ID: {obs['clause_id']}
Clause text: {obs['clause_text']}
Vendor stance: {vendor_stance}
Legal stance: {legal_stance}
Vendor response: {obs['vendor_response']}
Legal response: {obs['legal_response']}
Probe result: {obs.get('probe_result')}
Probes remaining: {probes_remaining}
Clauses agreed: {obs['clauses_agreed']} / {obs['clauses_total']}
Agreed clause IDs: {agreed_clauses}
Rounds remaining: {obs['rounds_remaining']}

=== LEARNED VENDOR VALUES (from probes this episode) ===
{json.dumps(probed_values, indent=2) if probed_values else "Nothing learned yet — consider probing"}

=== RECENT HISTORY ===
{chr(10).join(history[-6:])}

=== DECISION ===
Think step by step:
1. Is this clause already agreed? If yes, it will auto-advance — just ACCEPT.
2. Do I know the vendor's hidden value for this clause? Check learned values above.
3. If probes available and vendor value unknown: PROBE vendor first.
4. If I know vendor value: PROPOSE with that value in new_text.
5. If vendor=open and legal=approved: ACCEPT.
6. Never ESCALATE.

Respond with a single JSON action only."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=400,
        temperature=0.1,
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
    score = 0.01
    probed_values = {}  # clause_id -> what we learned from probing

    if _collector:
        _collector.start_episode(tier=tier)

    try:
        result = env_reset()
        obs = result["observation"]
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            try:
                action = get_action(client, model, obs, history, probed_values)
            except Exception as e:
                debug(f"get_action error: {e}")
                action = {"action_type": "ACCEPT", "clause_id": obs["clause_id"]}

            action["clause_id"] = action.get("clause_id") or obs["clause_id"]

            if done:
                rewards.append(0.01)
                steps_taken = step
                log_step(step, action["action_type"], 0.01, True)
                if _collector:
                    _collector.log_step(action, obs, 0.01)
                break

            try:
                result = env_step(action)
                obs_raw = result["observation"]

                # Track probe results to build up knowledge
                if action.get("action_type") == "PROBE" and obs_raw.get("probe_result"):
                    clause_id = action.get("clause_id", "unknown")
                    probed_values[clause_id] = obs_raw["probe_result"]

                obs = obs_raw
                reward_raw = result.get("reward")
                reward = float(reward_raw if reward_raw is not None else 0.01)
                reward = min(max(reward, 0.01), 0.99)
                done = result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.01
                done = False
                error = str(e)
                debug(f"step error: {e}")

            rewards.append(reward)
            steps_taken = step

            if _collector:
                _collector.log_step(action, obs, reward)

            meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
            log_step(step, action.get("action_type", "?"), reward, done, error)
            history.append(
                f"Step {step}: {action.get('action_type')} clause={action.get('clause_id')} "
                f"reward={reward:.4f} vendor={meta.get('vendor_stance','?')} legal={meta.get('legal_stance','?')}"
            )

            if done:
                break

        if not rewards:
            rewards = [0.01]

        total = sum(rewards)
        raw_ratio = total / len(rewards)
        score = safe_score(raw_ratio)
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

        debug(f"[SCORE] total={total} steps={len(rewards)} raw_ratio={raw_ratio:.6f} score={score}")
        debug(f"[PROBED] {probed_values}")

    except Exception as e:
        debug(f"tier error: {e}")
        if not rewards:
            rewards = [0.01]
        score = 0.01

    finally:
        log_end(success, steps_taken, rewards, score)
        if _collector:
            _collector.end_episode(final_score=score)

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