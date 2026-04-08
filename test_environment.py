from server.contractarena_environment import ContractarenaEnvironment
from models import ContractarenaAction, ActionType

def test_reset():
    env = ContractarenaEnvironment("easy")
    obs = env.reset()
    assert obs.clauses_total == 4
    assert obs.clauses_agreed == 0
    print("test_reset passed")

def test_probe_reward():
    env = ContractarenaEnvironment("easy")
    env.reset()
    r = env.step(ContractarenaAction(action_type=ActionType.PROBE, party="vendor", question="what?"))
    assert r.reward == 0.1
    print("test_probe_reward passed")

def test_propose_closes_clause():
    env = ContractarenaEnvironment("easy")
    env.reset()
    r = env.step(ContractarenaAction(action_type=ActionType.PROPOSE, clause_id="pricing", new_text="billed monthly"))
    assert r.clauses_agreed == 1
    print("test_propose_closes_clause passed")

def test_probe_budget_hard():
    env = ContractarenaEnvironment("hard")
    env.reset()
    for _ in range(3):
        env.step(ContractarenaAction(action_type=ActionType.PROBE, party="vendor", question="?"))
    r = env.step(ContractarenaAction(action_type=ActionType.PROBE, party="vendor", question="?"))
    assert "exhausted" in r.probe_result.lower()
    print("test_probe_budget passed")

def test_deterministic():
    env1 = ContractarenaEnvironment("easy")
    env2 = ContractarenaEnvironment("easy")
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert obs1.clause_id == obs2.clause_id
    print("test_deterministic passed")

if __name__ == "__main__":
    test_reset()
    test_probe_reward()
    test_propose_closes_clause()
    test_probe_budget_hard()
    test_deterministic()
    print("All tests passed")