import requests

BASE = "http://localhost:7860"

def demo():
    print("=== ContractArena Demo ===\n")

    r = requests.post(f"{BASE}/reset", json={})
    obs = r.json()["observation"]
    print(f"Episode started — tier: {obs['tier']}")
    print(f"First clause: {obs['clause_id']} — {obs['clause_text']}\n")

    print("Step 1: PROBE vendor")
    r = requests.post(f"{BASE}/step", json={"action": {
        "action_type": "PROBE", "clause_id": obs["clause_id"],
        "party": "vendor", "question": "What matters most to you?"
    }})
    result = r.json()
    print(f"  probe_result: {result['observation']['probe_result']}")
    print(f"  reward: {result['reward']}\n")

    print("Step 2: PROPOSE with hidden keyword")
    r = requests.post(f"{BASE}/step", json={"action": {
        "action_type": "PROPOSE",
        "clause_id": result["observation"]["clause_id"],
        "new_text": "Subscription fee: $500/month, billed monthly."
    }})
    result = r.json()
    print(f"  vendor: {result['observation']['vendor_response']}")
    print(f"  legal: {result['observation']['legal_response']}")
    print(f"  clauses_agreed: {result['observation']['clauses_agreed']}")
    print(f"  reward: {result['reward']}")
    print(f"  done: {result['done']}")

if __name__ == "__main__":
    demo()