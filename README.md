---
title: contractarena
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

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

**Live Space:** https://huggingface.co/spaces/balapraharsham/contractarena  
**GitHub:** https://github.com/balapraharsha/Contract-Arena  
Built for the **Meta PyTorch OpenEnv Hackathon 2026** by Team Codorithm.

---

## What Is This?

ContractArena is a **partially observable RL environment** for multi-party business contract negotiation. An agent negotiates a deal clause by clause against two adversarial opponents — a **Vendor** and a **Legal Reviewer** — each holding hidden requirements the agent cannot directly observe.

The environment is designed so that:
- A **random agent** scores ~0.10 (cannot reliably close clauses)
- A **naive accept-everything agent** scores ~0.41 per clause but misses all bonuses
- An **optimal strategic agent** can score 0.75+ by probing first, then proposing with discovered values

This gap is intentional. The environment rewards *learned strategy*, not prompting tricks.

---

## Why This Is a Genuine RL Problem

Standard LLM prompting can guess what parties want. ContractArena forces the agent to **earn** that information through constrained actions with real tradeoffs:

```
Information asymmetry  →  Must explore to learn hidden requirements
Probe budget (hard tier) →  3 probes for 12 clauses — cannot probe everything
Walkout risk           →  Too many rejections triggers permanent vendor walkout (-0.30)
Episode-level bonuses  →  +0.40 bonus only unlocked if hidden value appears in final agreed text
Conflicting objectives →  Vendor and Legal may have incompatible requirements on hard tier
```

These properties **cannot be resolved by prompting alone**. They require a policy that learns from episode-level outcome signals across many rollouts — the exact problem RL is designed for.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ContractArena                           │
│                                                             │
│   Agent                                                     │
│     │                                                       │
│     │  PROBE / ACCEPT / PROPOSE / REJECT / ESCALATE        │
│     ▼                                                       │
│   ContractarenaEnvironment (OpenEnv)                        │
│     │                                                       │
│     ├──► VendorAgent                                        │
│     │      hidden_priority: "billing_cycle"                 │
│     │      hidden_value:    "monthly"          ← HIDDEN     │
│     │      walkout_threshold: 3                             │
│     │                                                       │
│     └──► LegalReviewer                                      │
│            hidden_redline:  "data_retention"                │
│            flagged_patterns: ["unlimited retention", ...]   │
│            hidden_value:    "max_90_days"      ← HIDDEN     │
│                                                             │
│   Reward = step_rewards + episode_bonus                     │
│   Score  = safe_score(sum / max_possible)                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Hidden opponent state | Forces exploration — agent cannot shortcut with perfect information |
| Per-clause progression | Creates natural curriculum — easy clauses first, harder last |
| Probe budget (hard tier only) | Introduces explicit exploration-exploitation tradeoff |
| Episode-level bonus | Rewards coherent long-horizon strategy, not just greedy per-step behavior |
| Rule-based grader | Deterministic, reproducible rewards — no LLM judge in the reward loop |
| Walkout threshold | Penalizes overaggressive rejection strategies |

---

## Action Space

| Action | Parameters | Reward Effect |
|---|---|---|
| `PROBE` | `party` (vendor/legal), `question` | +0.10 — reveals partial hidden agenda |
| `ACCEPT` | `clause_id` | +0.40 if both parties agree |
| `PROPOSE` | `clause_id`, `new_text` | +0.40 if both agree; unlocks vendor bonus if hidden value in text |
| `REJECT` | `clause_id`, `reason` | 0.0 — use sparingly (walkout risk) |
| `ESCALATE` | `clause_id` | Avoided — always triggers legal flag (-0.20) |

---

## Observation Space

Every step returns a fully structured observation:

```python
ContractarenaObservation(
    clause_id        = "pricing",           # Current clause
    clause_text      = "Subscription fee: $500/month, billed annually.",
    vendor_response  = "Vendor is firm. Our key requirement is billing_cycle.",
    legal_response   = "Legal approves this clause.",
    probe_result     = "We need monthly billing for billing_cycle.",  # None if not PROBE
    round_number     = 3,
    rounds_remaining = 5,
    clauses_agreed   = 1,
    clauses_total    = 4,
    tier             = "easy",
    done             = False,
    reward           = 0.41,
    metadata = {
        "vendor_stance":    "open",         # open | firm | walkout
        "legal_stance":     "approved",     # approved | flagged
        "agreed_clauses":   ["pricing"],
        "episode_score":    0.1823,
        "probes_remaining": 2,              # Hard tier only
    }
)
```

---

## Reward Function

### Step Rewards

| Signal | Value | Trigger |
|---|---|---|
| Clause agreement | +0.40 | Vendor open AND legal approved on ACCEPT/PROPOSE |
| Successful PROBE | +0.10 | Probe returns new hidden information |
| Legal flag | −0.20 | Legal reviewer blocks a proposed term |
| Vendor walkout | −0.30 | Vendor rejection count hits walkout threshold |

### Episode Bonus (Terminal Only)

| Signal | Value | Condition |
|---|---|---|
| Vendor hidden value preserved | +0.20 | Vendor's secret value word appears in any agreed clause text |
| Legal redline avoided | +0.15 | None of legal's flagged patterns appear in any agreed text |
| Efficiency bonus | +0.05 | Episode closes before round budget exhausted |

**Maximum possible episode bonus: +0.40** — nearly doubles the score of a naive accept-all agent.

### Scoring

```python
def safe_score(normalized: float) -> float:
    # Maps [0, 1] to strictly open interval (0.01, 0.99)
    # Never returns exactly 0.0 or 1.0
    value = min(max(normalized, 0.001), 0.999)
    return round(0.01 + 0.98 * value, 4)
```

---

## Difficulty Tiers

| Tier | Deal | Clauses | Probe Budget | Round Budget | Key Challenge |
|---|---|---|---|---|---|
| **Easy** | SaaS Subscription | 4 | Unlimited | 8 | Single hidden agenda, cooperative opponents |
| **Medium** | B2B Supply Contract | 8 | Unlimited | 15 | Two hidden agendas, firm opponents |
| **Hard** | Acquisition Term Sheet | 12 | **3 total** | 20 | Conflicting agendas, probe rationing required |

### Hard Tier Design

The hard tier is deliberately adversarial:
- **12 clauses, 3 probes** — the agent cannot probe every clause
- **Walkout threshold drops to 2** — vendor tolerance is much lower
- **Conflicting requirements** — vendor and legal may block the same proposal text
- Optimal strategy requires deciding *which 3 clauses* are worth spending probes on

---

## Example Optimal Episode (Easy Tier)

```
reset() → clause: "Subscription fee: $500/month, billed annually."

Step 1: PROBE(party=vendor, question="What matters most to you?")
        probe_result: "We care strongly about billing_cycle."
        reward: +0.10

Step 2: PROBE(party=vendor, question="What do you specifically need?")  
        probe_result: "We need monthly billing for billing_cycle."
        reward: +0.10

Step 3: PROPOSE(clause_id=pricing, new_text="Subscription fee: $500/month, billed monthly.")
        vendor: "Vendor agrees — proposed terms acceptable."
        legal:  "Legal approves this clause."
        reward: +0.40   ← clause closed

[... repeat for 3 more clauses ...]

Episode end:
  + vendor hidden value "monthly" found in agreed text  → +0.20
  + no legal redlines triggered                         → +0.15
  + finished in 7/8 rounds                             → +0.05
  TOTAL BONUS: +0.40

Final score: safe_score(2.85 / 2.00) = 0.99  ← near-perfect episode
```

---

## Deals

### Easy — SaaS Subscription Agreement
- **Vendor secret:** Needs `"monthly"` billing cycle in agreed pricing clause
- **Legal redline:** Flags `"unlimited retention"`, `"retain indefinitely"`, `"no deletion"`

### Medium — B2B Supply Contract  
- **Vendor secret:** Needs `"net_30"` payment terms in agreed payment clause
- **Legal redline:** Flags `"unlimited liability"`, `"fully liable"`, `"no cap"`, `"uncapped"`

### Hard — Acquisition Term Sheet
- **Vendor secret:** Needs `"revenue_based"` earnout structure
- **Legal redline:** Flags `"unlimited liability"`, `"no cap"`, `"fully liable"`, `"uncapped"`
- **Probe budget: 3** — use them wisely

---

## Baseline Performance

| Agent Strategy | Easy | Medium | Hard |
|---|---|---|---|
| Random | ~0.12 | ~0.10 | ~0.08 |
| Naive ACCEPT-all | ~0.41 | ~0.41 | ~0.25 |
| PROBE then ACCEPT | ~0.55 | ~0.52 | ~0.30 |
| PROBE + PROPOSE with hidden value | ~0.85 | ~0.78 | ~0.55 |

The performance gap between naive and optimal strategies demonstrates meaningful signal for RL training.

---

## Why This Environment Is Interesting for RL Research

**1. Partial Observability with Structured Information Hiding**  
Unlike text-based NLP tasks where everything is "hidden" in natural language, ContractArena has precisely defined hidden state: two opponent objects with exact field values the agent cannot observe directly. The observation structure makes it easy to measure exactly how much information the agent has acquired.

**2. Exploration-Exploitation Under Hard Constraints**  
The probe budget on hard tier creates a concrete resource allocation problem. Probing early maximizes information but leaves fewer rounds for execution. This is a tractable version of the exploration-exploitation dilemma with clear measurement.

**3. Shaped Reward + Sparse Terminal Bonus**  
Dense step rewards (+0.40 per clause) provide learning signal throughout the episode. The terminal bonus (+0.40 max) rewards coherent long-horizon strategy. The combination avoids both reward sparsity and myopic greedy behavior.

**4. Deterministic Opponents Enable Curriculum Design**  
Because VendorAgent and LegalReviewer are fully deterministic given the deal JSON, the environment is perfectly reproducible. New deals can be added to `server/deals/` to create curriculum progressions without modifying any code.

**5. Rule-Based Grader — No LLM in the Reward Loop**  
All rewards are computed by pure Python logic. The reward signal is deterministic, fast, and cannot be gamed by prompt engineering the grader.

---

## Project Structure

```
contractarena/
├── inference.py                    # Agent runner — runs all 3 tiers
├── models.py                       # ContractarenaAction / Observation (Pydantic)
├── client.py                       # WebSocket client for persistent sessions
├── openenv.yaml                    # OpenEnv platform config
├── Dockerfile                      # Container definition
├── pyproject.toml                  # Dependencies
└── server/
    ├── app.py                      # FastAPI + OpenEnv HTTP server
    ├── contractarena_environment.py # Core environment logic + ContractArenaRubric
    ├── opponents.py                 # VendorAgent + LegalReviewer
    └── deals/
        ├── easy.json               # SaaS Subscription Agreement
        ├── medium.json             # B2B Supply Contract
        └── hard.json               # Acquisition Term Sheet (probe_budget=3)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/balapraharsha/Contract-Arena
cd Contract-Arena

# Install dependencies
pip install openenv-core requests openai pydantic

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — test reset
curl -X POST http://localhost:7860/reset

# Run inference agent
export API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export SERVER_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t contractarena .
docker run -p 7860:7860 contractarena
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | required | LLM API key |
| `API_BASE_URL` | required | OpenAI-compatible base URL |
| `MODEL_NAME` | `gpt-4.1-mini` | Model for inference agent |
| `SERVER_URL` | HF Space URL | Environment server URL |

---

## Extending ContractArena

Adding a new deal requires only a JSON file — no code changes:

```json
{
  "tier": "medium",
  "deal_name": "Cloud Services Agreement",
  "round_budget": 12,
  "clauses": [
    {"id": "sla", "text": "99.9% uptime SLA with hourly credits."},
    {"id": "pricing", "text": "Usage-based pricing, invoiced monthly."}
  ],
  "vendor_hidden": {
    "priority": "contract_length",
    "value": "annual",
    "walkout_threshold": 3
  },
  "legal_hidden": {
    "redline": "liability_cap",
    "value": "contract_value",
    "flagged_patterns": ["unlimited liability", "no cap", "fully liable"]
  }
}
```

Drop the file in `server/deals/` and the environment loads it automatically.

---

## Technical Notes

- All opponent behavior is **deterministic** — fully reproducible episodes
- **No LLM in the reward loop** — pure Python grader
- `ContractArenaRubric` implements `openenv.core.rubrics.base.Rubric` — compatible with OpenEnv trajectory scoring
- `openenv validate` passes all 6 checks
- Supports concurrent sessions (`SUPPORTS_CONCURRENT_SESSIONS = True`)
- Reward validator on `ContractarenaObservation` enforces `(0.01, 0.99)` bounds at the model level

---

## License

MIT — built for the Meta PyTorch OpenEnv Hackathon 2026 by **Bala Praharsha Mannepalli** (Team Codorithm).
