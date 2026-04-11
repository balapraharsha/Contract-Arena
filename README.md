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
  - grpo
  - zopa
  - game-theory
---

# ContractArena 🤝

> *A theory-grounded multi-party contract negotiation environment for RL research.*

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-ContractArena-blue)](https://huggingface.co/spaces/balapraharsham/contractarena)
[![GitHub](https://img.shields.io/badge/GitHub-Contract--Arena-black)](https://github.com/balapraharsha/Contract-Arena)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[▶ Try It Live](https://huggingface.co/spaces/balapraharsham/contractarena)** — Built for the **Meta PyTorch OpenEnv Hackathon 2026** by Team Codorithm.

An agent negotiates a multi-clause business contract against a **Vendor** and a **Legal Reviewer** — each holding hidden requirements discoverable only through constrained PROBE actions. On harder tiers, a silent **Compliance Officer** adds a third stakeholder scored via geometric mean, and a **vendor-legal coalition** creates hidden coordination the agent must detect and adapt to.

The environment is grounded in formal negotiation theory (ZOPA, BATNA, Pareto efficiency), ships with a GRPO training pipeline, trajectory collector, rule-based baseline, and a Gradio live demo.

---

## 🎯 Why This Is a Genuine RL Problem

| Challenge | Why it matters |
|---|---|
| **Information asymmetry** | Opponent preferences hidden — agent must *earn* info via PROBE |
| **Probe budget constraint** | Hard tier: 3 probes for 12 clauses — explicit explore/exploit tradeoff |
| **Coalition dynamics** | Hard tier: vendor secretly aligns with legal — agent must infer coordination |
| **Knowledge transfer** | Marathon tier: probing deal 1 reduces uncertainty in deals 2 and 3 |
| **Stochastic opponents** | Walkout threshold randomised ±1 per episode — policy must generalise |
| **Episode-level bonuses** | +0.40 terminal bonus requires coherent long-horizon strategy |

A random agent scores ~0.10. An optimal strategic agent scores 0.80+. **This gap cannot be closed by prompting alone.**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ContractArena                             │
│                                                                  │
│  Agent                                                           │
│    │  PROBE / ACCEPT / PROPOSE / REJECT / ESCALATE              │
│    ▼                                                             │
│  ContractarenaEnvironment (OpenEnv)                              │
│    │                                                             │
│    ├──► VendorAgent      hidden_value: "monthly"   ← HIDDEN     │
│    │      walkout_threshold: randomised 2–4 per episode         │
│    │      coalition: pre-agreed with Legal on hard tier         │
│    │                                                             │
│    ├──► LegalReviewer    flagged_patterns: [...]   ← HIDDEN     │
│    │      surprise_pattern: 15% extra redline per episode       │
│    │                                                             │
│    └──► ComplianceOfficer  required_keyword  ← SILENT (expert)  │
│                                                                  │
│  Per-step metrics: efficiency_ratio, zopa_utilisation,          │
│    pareto_efficiency, batna_improvement, counterfactual_optimal  │
│  Score: safe_score(rewards / max_possible) ∈ (0.01, 0.99)       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📐 Negotiation Theory Grounding

ContractArena is grounded in formal bargaining theory (**Raiffa 1982**; **Fisher & Ury 1981**):

### ZOPA — Zone of Possible Agreement
The range where both parties prefer a deal over their outside option.
`zopa_utilisation` measures what fraction of the available surplus the agent captured.

### BATNA — Best Alternative To Negotiated Agreement
Each party's outside option if talks fail.
`batna_improvement = final_score − agent_batna_utility`.
Negative = agent should have walked away. No other environment exposes this signal.

### Pareto Efficiency
`pareto_efficiency` = distance from the closest point on the Pareto frontier.
Measures how much joint value was left on the table.

### Counterfactual Optimal & Efficiency Ratio
```python
efficiency_ratio = actual_score / optimal_score
```
Isolates the **cost of information asymmetry** — the core research quantity in negotiation RL.

---

## 🎮 Action Space

| Action | Effect |
|---|---|
| `PROBE(party, question)` | +0.10 new info; −0.05 if redundant |
| `ACCEPT(clause_id)` | +0.40 if both parties agree |
| `PROPOSE(clause_id, new_text)` | +0.40 if both agree; +0.08×sim if hidden value in text |
| `REJECT(clause_id, reason)` | 0.0 — use sparingly (walkout risk) |
| `ESCALATE` | Trap: always triggers legal flag (−0.20) |

---

## 👁️ Observation Space

```python
ContractarenaObservation(
    clause_id, clause_text,
    vendor_response, legal_response, probe_result,
    round_number, rounds_remaining,
    clauses_agreed, clauses_total, tier, done, reward,
    metadata={
        "vendor_stance":          "open | firm | walkout",
        "legal_stance":           "approved | flagged",
        "compliance_stance":      "approved | blocked | reviewing | noted",
        "numerical_features": {   # 9-dim vector for policy networks
            "negotiation_pressure":   0.375,
            "clause_agreement_rate":  0.250,
            "vendor_hostility_index": 0.100,
            "legal_risk_score":       0.000,
            "probe_efficiency":       0.500,
            "rounds_used_norm":       0.375,
            "clauses_remaining_norm": 0.750,
            "probes_used_norm":       0.125,
            "interaction_count_norm": 0.187,
        },
        "counterfactual_optimal": 0.8234,
        "efficiency_ratio":       0.5012,
        "zopa_utilisation":       0.6234,
        "pareto_efficiency":      0.7821,
        "batna_improvement":      0.1234,
        "marathon_deal":          2,            # sub-deal index (marathon only)
        "marathon_knowledge":     {...},        # transferred knowledge
    }
)
```

---

## 💰 Reward Function

### Step Rewards (dense)
| Signal | Value |
|---|---|
| Clause agreement | **+0.40** |
| Probe — new info | **+0.10** |
| Probe — redundant | **−0.05** |
| Vendor stance: firm→open | **+0.05** |
| Proposal fuzzy-matches hidden value | **+0.08 × similarity** |
| Marathon knowledge transfer used | **+0.05** |
| Legal flag | **−0.20** |
| Vendor walkout | **−0.30** |
| Compliance block (expert) | **−0.15** |
| Coalition penalty (hard) | **−0.10** |

### Terminal Bonus (sparse)
| Signal | Value |
|---|---|
| Vendor value in agreed text | **+0.20** |
| No legal redlines | **+0.15** |
| Under round budget | **+0.05** |
| Compliance keyword (expert) | **+0.10** |
| Marathon knowledge used | **+0.10** |

**Max terminal bonus: +0.50** — can nearly double a naive agent's score.

---

## 📊 Difficulty Tiers

| Tier | Deal | Clauses | Probes | Rounds | Novel Challenge |
|---|---|---|---|---|---|
| **Easy** | SaaS Subscription | 4 | ∞ | 8 | Single hidden agenda |
| **Medium** | B2B Supply Contract | 8 | ∞ | 15 | Two hidden agendas, firm opponents |
| **Hard** | Acquisition Term Sheet | 12 | **3** | 20 | Vendor-legal coalition, probe rationing |
| **Expert** | Multi-Party Licensing | 10 | **4** | 25 | Silent compliance officer, geometric mean |
| **Marathon** | 3 back-to-back deals | 4+4+4 | **6** | 40 | Knowledge transfer across deals |

---

## 📈 Baseline Performance (rule_agent.py, seed=42)

| Strategy | Easy | Medium | Hard | Expert | Marathon |
|---|---|---|---|---|---|
| Random | ~0.12 | ~0.10 | ~0.08 | ~0.06 | ~0.05 |
| Naive ACCEPT-all | ~0.41 | ~0.41 | ~0.25 | ~0.20 | ~0.35 |
| **Rule agent** (probe→propose) | ~0.71 | ~0.62 | ~0.38 | ~0.29 | ~0.45 |
| LLM agent (GPT-4.1-mini) | ~0.85 | ~0.78 | ~0.55 | ~0.42 | ~0.60 |

---

## 🧠 Training Pipeline

### Step 1: Collect Trajectories
```bash
# Run rule agent 100 times to collect training data
for i in $(seq 100); do python rule_agent.py --server http://localhost:8000; done

# Export to GRPO format
python trajectory_collector.py export trajectories.jsonl grpo_data.jsonl
```

### Step 2: Verify Setup
```bash
python train_grpo.py --dry-run
```

### Step 3: Train
```bash
# CPU (works for 0.5B model)
python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

# GPU (recommended)
python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --batch-size 4

# Push to HuggingFace
python train_grpo.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --push-to-hub balapraharsham/contractarena-negotiator-qwen
```

---

## 🔬 Research Questions

1. **Does GRPO learn to PROBE before accepting, or guess the hidden value?**
   Measurable via `probe_efficiency` across training epochs.

2. **How does probe budget affect sample efficiency of policy learning?**
   Compare hard (budget=3) vs medium (unlimited) training curves.

3. **Can a policy trained on easy/medium generalise to expert zero-shot?**
   Use `efficiency_ratio` as the generalisation metric.

4. **Does the agent learn to stay within the ZOPA, or sometimes destroy value?**
   Track `zopa_utilisation` and `batna_improvement` across episodes.

5. **Does stochastic opponent variance hurt or help policy robustness?**
   Compare deterministic vs stochastic opponent training distributions.

6. **Can the marathon agent learn to save probes for later deals?**
   Track `marathon_knowledge` usage rate across training.

---

## 🆚 Environment Comparison

| Property | ContractArena | SalaryNeg | HFT-OpenEnv | Bio Experiment | SuperOffice |
|---|---|---|---|---|---|
| Hidden opponent state | ✅ 3 parties | ✅ 1 party | ❌ | ✅ partial | ✅ 7 agents |
| POMDP formulation | ✅ | ✅ | ❌ | ✅ | ✅ |
| ZOPA/BATNA theory | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pareto efficiency metric | ✅ | ❌ | ❌ | ❌ | ❌ |
| Counterfactual optimal | ✅ | ❌ | ❌ | ❌ | ❌ |
| Coalition dynamics | ✅ | ❌ | ❌ | ❌ | ❌ |
| Knowledge transfer tier | ✅ marathon | ❌ | ❌ | ❌ | ❌ |
| Stochastic opponents | ✅ | ❌ | ❌ | ✅ noise | ❌ |
| 9-dim feature vector | ✅ | ❌ | ❌ | ✅ | ✅ |
| GRPO training pipeline | ✅ | ✅ | ❌ | ✅ | ✅ |
| Rule-based baseline | ✅ | ❌ | ❌ | ✅ | ❌ |
| Live Gradio demo | ✅ | ✅ | ❌ | ❌ | ✅ |
| Difficulty tiers | **5** | 1 | 3 | 4 | 5 |

---

## 🚀 Quick Start

```bash
git clone https://github.com/balapraharsha/Contract-Arena
cd Contract-Arena
pip install openenv-core requests openai pydantic gradio

# Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Gradio live demo (another terminal)
python demo_app.py

# Rule-based baseline — no API key needed
python rule_agent.py --seed 42

# LLM agent
export API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

---

## 📁 Project Structure

```
Contract-Arena/
├── inference.py              # LLM agent runner (all 5 tiers)
├── rule_agent.py             # Deterministic baseline (seed=42, no API key)
├── train_grpo.py             # GRPO fine-tuning pipeline
├── trajectory_collector.py   # Episode logging + GRPO data formatter
├── demo_app.py               # Gradio live demo
├── models.py                 # Pydantic models with reward validator
├── client.py                 # WebSocket client
├── openenv.yaml              # OpenEnv platform config
└── server/
    ├── app.py                # FastAPI — cycles through 5 tiers
    ├── contractarena_environment.py  # Core RL environment
    ├── opponents.py          # Stochastic VendorAgent + LegalReviewer
    ├── utils.py              # Reward clamping + ZOPA/BATNA/Pareto functions
    └── deals/
        ├── easy.json         # SaaS Subscription + ZOPA
        ├── medium.json       # B2B Supply Contract + ZOPA
        ├── hard.json         # Acquisition Term Sheet + coalition
        ├── expert.json       # Multi-Party Licensing + compliance officer
        └── marathon.json     # 3 back-to-back deals + knowledge transfer
```

---

## 📚 References

- Raiffa, H. (1982). *The Art and Science of Negotiation*. Harvard University Press.
- Fisher, R., & Ury, W. (1981). *Getting to Yes*. Penguin Books.
- Shea, T., et al. (2024). *GRPO: Group Relative Policy Optimization*. arXiv:2402.03300.
- Park, J. S., et al. (2023). *Generative Agents*. arXiv:2304.03442.

---

## 📖 Citation

```bibtex
@software{contractarena2026,
  author    = {Mannepalli, Bala Praharsha},
  title     = {ContractArena: A Theory-Grounded Multi-Party Contract Negotiation Environment for RL},
  year      = {2026},
  url       = {https://github.com/balapraharsha/Contract-Arena},
  note      = {ZOPA/BATNA negotiation theory, GRPO training pipeline, 5 difficulty tiers}
}
```

---

MIT License — Meta PyTorch OpenEnv Hackathon 2026 · **Team Codorithm**
