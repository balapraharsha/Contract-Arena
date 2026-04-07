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

# Contractarena Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Contractarena environment is through the `ContractarenaEnv` class:

```python
from contractarena import ContractarenaAction, ContractarenaEnv

try:
    # Create environment from Docker image
    contractarenaenv = ContractarenaEnv.from_docker_image("contractarena-env:latest")

    # Reset
    result = contractarenaenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = contractarenaenv.step(ContractarenaAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    contractarenaenv.close()
```

That's it! The `ContractarenaEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t contractarena-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**ContractarenaAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**ContractarenaObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Contractarena environment server running, you can connect directly:

```python
from contractarena import ContractarenaEnv

# Connect to existing server
contractarenaenv = ContractarenaEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = contractarenaenv.reset()
result = contractarenaenv.step(ContractarenaAction(message="Hello!"))
```

Note: When connecting to an existing server, `contractarenaenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from contractarena import ContractarenaAction, ContractarenaEnv

# Connect with context manager (auto-connects and closes)
with ContractarenaEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(ContractarenaAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    ContractarenaEnvironment,  # Pass class, not instance
    ContractarenaAction,
    ContractarenaObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from contractarena import ContractarenaAction, ContractarenaEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with ContractarenaEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(ContractarenaAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/contractarena_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
contractarena/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # ContractarenaEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── contractarena_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
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