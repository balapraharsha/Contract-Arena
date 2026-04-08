---
title: contractarena
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - negotiation
  - multi-agent
  - information-asymmetry
---

# ContractArena 🤝

> *A multi-party contract negotiation environment where information asymmetry is the core challenge.*

## 🚀 Try It Live

**[▶ Open Live Demo](https://huggingface.co/spaces/balapraharsham/contractarena)**

An agent negotiates a business contract clause-by-clause against a **Vendor** and a **Legal Reviewer** — each holding hidden requirements the agent must discover through constrained PROBE actions. On the expert tier, a silent **Compliance Officer** adds a third stakeholder whose satisfaction uses geometric mean scoring, meaning one bad compliance hit tanks the whole episode score. The agent must decide when to spend scarce probe budget, how to word proposals to include hidden values, and how to avoid triggering legal redlines — all under a fixed round budget.

**GitHub:** https://github.com/balapraharsha/Contract-Arena  
Built for the **Meta PyTorch OpenEnv Hackathon 2026** by Team Codorithm.

---

## What Is This?

ContractArena is a **partially observable RL environment** for multi-party business contract negotiation. An agent negotiates a deal clause by clause against two adversarial opponents — a **Vendor** and a **Legal Reviewer** — each holding hidden requirements the agent cannot directly observe.

The environment is designed so that:
- A **random agent** scores ~0.10 (cannot reliably close clauses)
- A **rule-based baseline** scores ~0.62 (see `rule_agent.py` and baseline table below)
- An **optimal strategic agent** can score 0.75+ by probing first, then proposing with discovered values

This gap is intentional. The environment rewards *learned strategy*, not prompting tricks.

---

## Baseline Scores (seed=42, `rule_agent.py`)

| Agent | Easy | Medium | Hard | Expert | Mean |
|---|---|---|---|---|---|
| Random | ~0.12 | ~0.10 | ~0.08 | ~0.06 | ~0.09 |
| Naive ACCEPT-all | ~0.41 | ~0.41 | ~0.25 | ~0.18 | ~0.31 |
| **Rule-based (`rule_agent.py`)** | **~0.71** | **~0.62** | **~0.38** | **~0.29** | **~0.50** |
| LLM agent (`inference.py`) | ~0.85 | ~0.78 | ~0.55 | ~0.42 | ~0.65 |

Run the reproducible baseline yourself:
```bash
python rule_agent.py --seed 42 --server http://localhost:7860
```

---

## Why This Is a Genuine RL Problem

Standard LLM prompting can guess what parties want. ContractArena forces the agent to **earn** that information through constrained actions with real tradeoffs:

```
Information asymmetry  →  Must explore to learn hidden requirements
Probe budget (hard/expert) →  3–4 probes for 10–12 clauses — cannot probe everything
Walkout risk           →  Too many rejections triggers permanent vendor walkout (-0.30)
Stochastic opponents   →  Walkout threshold varies 2–4 per episode; legal adds surprise patterns
Episode-level bonuses  →  +0.40 bonus only unlocked if hidden value appears in final agreed text
Expert geometric mean  →  Compliance score multiplied with negotiation score; one failure cascades
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ContractArena                            │
│                                                                 │
│   Agent                                                         │
│     │  PROBE / ACCEPT / PROPOSE / REJECT / ESCALATE            │
│     ▼                                                           │
│   ContractarenaEnvironment (OpenEnv)                            │
│     │                                                           │
│     ├──► VendorAgent          (stochastic: walkout 2–4/ep)     │
│     │      hidden_priority: "billing_cycle"                     │
│     │      hidden_value:    "monthly"          ← HIDDEN         │
│     │                                                           │
│     ├──► LegalReviewer        (stochastic: +surprise patterns) │
│     │      hidden_redline:    "data_retention"                  │
│     │      flagged_patterns:  ["unlimited retention", ...]      │
│     │                                                           │
│     └──► ComplianceOfficer    (expert tier only, silent)        │
│            required_keyword:  "quarterly_audit"   ← HIDDEN      │
│            geometric mean scoring: score = √(neg × compliance)  │
│                                                                 │
│   Reward = step_rewards (dense) + episode_bonus (sparse)        │
│   Score  = safe_score(sum / max_possible)   [see utils.py]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reward Function

### Step Rewards (dense — every step has signal)

| Signal | Value | Trigger |
|---|---|---|
| Clause agreement | +0.40 | Vendor open AND legal approved on ACCEPT/PROPOSE |
| Successful PROBE | +0.10 | Probe returns new (non-redundant) information |
| Redundant PROBE | −0.05 | Probe returns information already known |
| Vendor stance shifts firm→open | +0.05 | Partial progress shaping signal |
| Proposal fuzzy-matches hidden value | +0.08 × similarity | Gradient toward correct wording |
| Legal flag | −0.20 | Legal reviewer blocks a proposed term |
| Vendor walkout | −0.30 | Vendor rejection count hits walkout threshold |
| Compliance block (expert) | −0.15 | Compliance officer blocks a term |

### Episode Bonus (terminal only)

| Signal | Value | Condition |
|---|---|---|
| Vendor hidden value preserved | +0.20 | Vendor's secret value word appears in any agreed clause |
| Legal redline avoided | +0.15 | None of legal's flagged patterns in any agreed text |
| Efficiency bonus | +0.05 | Episode closes before round budget exhausted |
| Compliance keyword (expert) | +0.10 | Compliance required keyword in agreed text |

### Reward Clamping

All rewards are clamped to the open interval **(0.01, 0.99)** rather than [0, 1].

**Why not [0, 1]?** Two reasons: (1) OpenEnv platform validators treat exactly-0 as "agent never acted" and exactly-1 as "perfect episode" — both cause edge-case logging noise. (2) The `safe_score` mapping (`0.01 + 0.98 * x`) is only well-behaved on the open interval; hard boundaries produce identical gradient signal to nearby values and confuse reward normalisation in PPO. This design decision is intentional and consolidated in `server/utils.py`.

```python
# server/utils.py — single source of truth for reward clamping
def clamp(value: float) -> float:
    """Hard clamp to open interval (0.01, 0.99)."""
    return round(min(max(float(value), 0.01), 0.99), 4)

def safe_score(value: float) -> float:
    """Map arbitrary float to strict open interval (0.01, 0.99)."""
    value = min(max(float(value), 0.001), 0.999)
    return round(0.01 + 0.98 * value, 4)
```

---

## Difficulty Tiers

| Tier | Deal | Clauses | Probe Budget | Round Budget | Key Challenge |
|---|---|---|---|---|---|
| **Easy** | SaaS Subscription | 4 | Unlimited | 8 | Single hidden agenda, cooperative opponents |
| **Medium** | B2B Supply Contract | 8 | Unlimited | 15 | Two hidden agendas, firm opponents |
| **Hard** | Acquisition Term Sheet | 12 | **3 total** | 20 | Conflicting agendas, probe rationing |
| **Expert** | Multi-Party Licensing | 10 | **4 total** | 25 | Three stakeholders, geometric mean scoring |

### Expert Tier Design

The expert tier adds a **silent Compliance Officer** with a hidden required keyword. The agent must:
1. Discover the compliance requirement via `PROBE party=compliance`
2. Include the keyword in at least one agreed clause
3. Avoid triggering compliance block patterns

The final score is `√(negotiation_score × compliance_score)` — a geometric mean that ensures one bad compliance failure cascades across the whole episode score, even if negotiation went well.

---

## Enriched Observation Space (9-dim numerical features)

Every observation now includes a normalised numerical feature vector alongside text, making the environment suitable for policy networks that combine text and vector inputs:

```python
metadata["numerical_features"] = {
    "negotiation_pressure":   0.42,  # rounds_used / round_budget
    "clause_agreement_rate":  0.50,  # clauses_agreed / clauses_total
    "vendor_hostility_index": 0.20,  # rejection_count / total_interactions
    "legal_risk_score":       0.00,  # flagged_patterns_matched / total_patterns
    "probe_efficiency":       0.50,  # clauses_agreed / probes_used
    "rounds_used_norm":       0.42,
    "clauses_remaining_norm": 0.50,
    "probes_used_norm":       0.13,
    "interaction_count_norm": 0.21,
}
```

---

## Stochastic Opponents

Opponents are now stochastic per-episode to prevent policy memorisation:

- **VendorAgent**: walkout threshold varies ±1 per episode (range 2–4); may mention a random secondary concern during probe
- **LegalReviewer**: 15% chance of adding one surprise flagged pattern per episode (drawn from a fixed pool); agent can discover it via probe

This is ~20 lines of code in `server/opponents.py` but makes the environment require genuine generalisation rather than pattern matching.

---

## Trajectory Collection & GRPO Data

ContractArena ships a `TrajectoryCollector` that logs every episode to JSONL and exports GRPO-compatible training data:

```python
from trajectory_collector import TrajectoryCollector

collector = TrajectoryCollector("trajectories.jsonl")
collector.start_episode(tier="easy", episode_id="ep_001")
collector.log_step(action, observation, reward)
collector.end_episode(final_score=0.82)

# Export GRPO dataset (compatible with trl.GRPOTrainer)
collector.export_grpo("grpo_dataset.jsonl")

# Stats
print(collector.stats())
# {"episodes": 10, "mean_score": 0.71, "by_tier": {"easy": 0.82, "hard": 0.55}}
```

GRPO output format:
```json
{
  "prompt": "Clause: pricing — Subscription fee...\nVendor: We care about billing_cycle [firm]...",
  "completions": ["{\"action_type\": \"PROPOSE\", \"new_text\": \"...monthly...\"}"],
  "rewards": [0.41],
  "metadata": {"tier": "easy", "numerical_features": {...}}
}
```

---

## Project Structure

```
contractarena/
├── inference.py                    # LLM agent runner — runs all 4 tiers
├── rule_agent.py                   # Deterministic baseline (reproducible, seed=42)
├── trajectory_collector.py         # Episode logging + GRPO data export
├── models.py                       # ContractarenaAction / Observation (Pydantic)
├── client.py                       # WebSocket client for persistent sessions
├── openenv.yaml                    # OpenEnv platform config
├── Dockerfile                      # Container definition
├── pyproject.toml                  # Dependencies
└── server/
    ├── app.py                      # FastAPI + OpenEnv HTTP server
    ├── contractarena_environment.py # Core environment logic + ContractArenaRubric
    ├── opponents.py                 # VendorAgent + LegalReviewer (stochastic)
    ├── utils.py                    # Reward clamping + numerical feature builders
    └── deals/
        ├── easy.json               # SaaS Subscription Agreement
        ├── medium.json             # B2B Supply Contract
        ├── hard.json               # Acquisition Term Sheet (probe_budget=3)
        └── expert.json             # Multi-Party Licensing (3 stakeholders)
```

---

## Quick Start

```bash
git clone https://github.com/balapraharsha/Contract-Arena
cd Contract-Arena
pip install openenv-core requests openai pydantic

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run reproducible rule-based baseline (no API key needed)
python rule_agent.py --seed 42 --server http://localhost:7860

# Run LLM inference agent
export API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export SERVER_URL=http://localhost:7860
python inference.py

# Export GRPO training data from collected trajectories
python trajectory_collector.py export trajectories.jsonl grpo_dataset.jsonl
python trajectory_collector.py stats trajectories.jsonl
```

---

## Technical Notes

- Reward clamping consolidated in `server/utils.py` — single source of truth, no duplication
- Opponents are stochastic per-episode (walkout threshold 2–4, 15% surprise legal pattern)
- `numerical_features` dict in metadata enables vector-based RL policy inputs
- Partial reward shaping (firm→open transition, fuzzy match) provides dense gradient signal
- Expert tier uses geometric mean scoring across negotiation + compliance dimensions
- `TrajectoryCollector` logs JSONL compatible with GRPO/trl training pipelines
- All opponent behaviour seeded via Python `random` — pass `--seed` to `rule_agent.py` for reproducibility
- `openenv validate` passes all checks; `SUPPORTS_CONCURRENT_SESSIONS = True`

---

## License

MIT — built for the Meta PyTorch OpenEnv Hackathon 2026 by **Bala Praharsha Mannepalli** (Team Codorithm).
