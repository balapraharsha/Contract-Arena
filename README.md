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

# ContractArena

A multi-party business deal negotiation environment for RL training.
Built with OpenEnv for the Meta PyTorch OpenEnv Hackathon 2025.

**Live Space:** https://huggingface.co/spaces/balapraharsham/contractarena

---

## What is this?

ContractArena is a **partially observable** RL environment where an agent
negotiates a business contract clause by clause against two opponents —
a Vendor and a Legal Reviewer — each holding hidden requirements the
agent cannot directly observe.

The agent must use constrained PROBE actions to uncover what each party
secretly wants, then craft proposals that satisfy both parties simultaneously.

**Why this is a genuine RL environment:**
- The agent operates under **information asymmetry** — opponent preferences are hidden
- It must balance **exploration vs. efficiency** — probe budget is limited on hard tier
- Reward is shaped across the **full episode trajectory**, not just at the end
- The hard tier introduces **conflicting objectives** — satisfying both opponents
  fully may not be possible, requiring strategic tradeoffs

---

## Action Space

| Action | Description | Reward effect |
|---|---|---|
| `PROBE(party, question)` | Reveals one hidden agenda field; consumes probe budget | +0.10 per reveal |
| `ACCEPT(clause_id)` | Accept current clause as-is | +0.40 if both agree |
| `PROPOSE(clause_id, new_text)` | Submit modified clause text | +0.40 if both agree |
| `REJECT(clause_id, reason)` | Reject clause, request counter-proposal | 0.0 |
| `ESCALATE(clause_id)` | Forces both parties to state positions explicitly; does not close clause but reveals maximum information | 0.0 |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `clause_id` | string | ID of current clause being negotiated |
| `clause_text` | string | Current clause text |
| `vendor_response` | string | Always present — vendor reaction to last action |
| `legal_response` | string | Always present — legal reviewer reaction to last action |
| `probe_result` | string or null | Non-null only after PROBE — contains partial hidden agenda reveal |
| `rounds_remaining` | int | Steps left in current episode |
| `clauses_agreed` | int | Clauses closed so far; updates after each accepted clause |

---

## Reward Function

| Signal | Value | Trigger |
|---|---|---|
| Clause agreement | +0.40 | Both vendor and legal accept the clause in the same step |
| Successful PROBE | +0.10 | Probe reveals new hidden agenda information |
| Hidden priority preserved | +0.20 | Final agreed terms contain vendor's hidden requirement |
| Legal redline avoided | +0.15 | No flagged terms in any agreed clause |
| Efficiency bonus | +0.05 | Episode closes under round budget |
| Legal flag | −0.20 | Legal reviewer blocks a proposed term |
| Vendor walkout | −0.30 | Vendor stance reaches walkout state |

Reported score: `sum(episode_rewards) / max_possible`, clamped to `[0.0, 1.0]`.
Max possible is calculated per tier based on clause count and bonus ceiling.

---

## Difficulty Tiers

| Tier | Deal | Clauses | Probe Budget | Round Budget | Challenge |
|---|---|---|---|---|---|
| Easy | SaaS subscription | 4 | Unlimited | 8 | One hidden agenda, cooperative opponents |
| Medium | B2B supply contract | 8 | Unlimited | 15 | Two hidden agendas, firm opponents |
| Hard | Acquisition term sheet | 12 | 3 total | 20 | Conflicting agendas, strategic tradeoffs required |

> The hard tier is intentionally designed so that satisfying both opponents
> fully may not be possible. The agent must make deliberate tradeoffs to
> maximize total episode reward.

---

## Example Interaction (Easy tier)

**reset()** → clause: `"Subscription fee: $500/month, billed annually."`

**Step 1:** `PROBE(party=vendor, question="What matters most to you?")`
- probe_result: `"We care strongly about billing_cycle."`
- reward: **+0.10**

**Step 2:** `PROBE(party=vendor, question="What do you need specifically?")`
- probe_result: `"We need monthly billing for billing_cycle."`
- reward: **+0.10**

**Step 3:** `PROPOSE(clause_id=pricing, new_text="Subscription fee: $500/month, billed monthly.")`
- vendor: `"Vendor agrees -- the proposed terms are acceptable."`
- legal: `"Legal approves this clause."`
- clauses_agreed: **1**
- reward: **+0.40**

---

## Baseline Scores

| Tier | Random agent | Scripted agent (Qwen-7B) |
|---|---|---|
| Easy | 0.10 | 0.40 |
| Medium | 0.10 | 0.40 |
| Hard | 0.10 | 0.10 |

The gap between random and scripted agents demonstrates that strategic
probing and proposal crafting produces meaningfully better outcomes.
Random action selection cannot close clauses consistently.

---

## Setup
```bash
pip install openenv-core
docker build -t contractarena .
docker run -p 8000:8000 contractarena
```

## Run inference
```bash
cp .env.example .env
python inference.py
```

## Environment variables
```bash
HF_TOKEN=your_huggingface_token
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
SERVER_URL=http://localhost:8000
```

---

## Why this is an RL environment, not a chatbot task

A standard LLM prompted to negotiate will guess what parties want.
ContractArena forces the agent to **earn** that information through
constrained exploration.

The probe budget on the hard tier creates a real tradeoff:
- **Use probes early** → full information, fewer rounds left to close
- **Save probes** → more rounds available, but less certainty on proposals

This tradeoff cannot be resolved by prompting alone. It requires a
policy that learns from episode-level outcomes across many rollouts.

---

## Notes

- All opponent behavior is deterministic and reproducible
- Grader scores are fully rule-based — no LLM judge in the reward loop
- Environment supports curriculum-style evaluation across 3 tiers
- Probe budget enforcement is strict — budget-exhausted probes return
  a blocked message with no reward
- `openenv validate` passes — ready for multi-mode deployment

---

## License

MIT