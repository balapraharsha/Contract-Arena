import json
import re
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ContractarenaAction, ContractarenaObservation
    from .opponents import VendorAgent, LegalReviewer
except ImportError:
    from models import ContractarenaAction, ContractarenaObservation
    from server.opponents import VendorAgent, LegalReviewer

DEALS_DIR = Path(__file__).parent / "deals"


def load_deal(tier: str) -> dict:
    path = DEALS_DIR / f"{tier}.json"
    with open(path) as f:
        return json.load(f)


class ContractarenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, tier: str = "easy"):
        self._tier = tier
        self._deal = load_deal(tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_opponents()
        self._reset_episode()

    def _init_opponents(self):
        vh = self._deal["vendor_hidden"]
        lh = self._deal["legal_hidden"]
        self._vendor = VendorAgent(
            hidden_priority=vh["priority"],
            hidden_value=vh["value"],
            walkout_threshold=vh.get("walkout_threshold", 3),
        )
        self._legal = LegalReviewer(
            hidden_redline=lh["redline"],
            hidden_value=lh["value"],
            flagged_patterns=lh["flagged_patterns"],
        )

    def _reset_episode(self):
        self._clauses = self._deal["clauses"]
        self._clause_index = 0
        self._agreed = {}
        self._round_budget = self._deal["round_budget"]
        self._probe_budget = self._deal.get("probe_budget", 999)
        self._probes_used = 0
        self._rounds_used = 0
        self._episode_rewards = []
        self._vendor_stance = "open"
        self._legal_stance = "approved"
        self._vendor.reset()
        self._legal.reset()

    def reset(self) -> ContractarenaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_episode()
        clause = self._clauses[0]
        return ContractarenaObservation(
            clause_id=clause["id"],
            clause_text=clause["text"],
            vendor_response="Vendor is ready to negotiate.",
            legal_response="Legal is ready to review.",
            probe_result=None,
            round_number=0,
            rounds_remaining=self._round_budget,
            clauses_agreed=0,
            clauses_total=len(self._clauses),
            tier=self._tier,
            done=False,
            reward=0.0,
        )

    def step(self, action: ContractarenaAction) -> ContractarenaObservation:
        self._state.step_count += 1
        self._rounds_used += 1

        clause = self._clauses[self._clause_index]
        clause_id = clause["id"]
        new_text = action.new_text or clause["text"]
        action_type = action.action_type.value

        # Get opponent responses
        vendor_resp, vendor_stance = self._vendor.respond(
            action_type, clause_id, new_text
        )
        legal_resp, legal_stance = self._legal.review(
            action_type, clause_id, new_text
        )
        self._vendor_stance = vendor_stance
        self._legal_stance = legal_stance

        # Handle probe result
        probe_result = None
        if action_type == "PROBE":
            if self._probes_used >= self._probe_budget:
                probe_result = "Probe budget exhausted — no more information available."
                vendor_resp = "No further information."
                legal_resp = "No further information."
                vendor_stance = self._vendor_stance
                legal_stance = self._legal_stance
            else:
                self._probes_used += 1
                party = (action.party or "vendor").lower()
                if party == "vendor":
                    probe_result = vendor_resp
                else:
                    probe_result = legal_resp

        # Calculate reward
        reward = self._calculate_reward(
            action_type, vendor_stance, legal_stance, probe_result
        )

        # If both agree, mark clause as agreed and advance
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                self._agreed[clause_id] = new_text
                self._clause_index += 1

        self._episode_rewards.append(reward)

        # Check done conditions
        all_agreed = len(self._agreed) == len(self._clauses)
        walkout = vendor_stance == "walkout"
        out_of_rounds = self._rounds_used >= self._round_budget
        done = all_agreed or walkout or out_of_rounds

        # Add final bonus on done
        if done:
            bonus = self._calculate_final_bonus()
            reward += bonus
            self._episode_rewards[-1] += bonus

        # Final score
        max_possible = len(self._clauses) * 0.40 + 0.40
        raw = sum(self._episode_rewards)
        score = round(min(max(raw / max_possible, 0.0), 1.0), 4)

        # Advance to next clause if available
        if not done and self._clause_index < len(self._clauses):
            next_clause = self._clauses[self._clause_index]
        else:
            next_clause = clause

        return ContractarenaObservation(
            clause_id=next_clause["id"],
            clause_text=next_clause["text"],
            vendor_response=vendor_resp,
            legal_response=legal_resp,
            probe_result=probe_result,
            round_number=self._rounds_used,
            rounds_remaining=self._round_budget - self._rounds_used,
            clauses_agreed=len(self._agreed),
            clauses_total=len(self._clauses),
            tier=self._tier,
            done=done,
            reward=round(score if done else reward, 4),
            metadata={
                "vendor_stance": vendor_stance,
                "legal_stance": legal_stance,
                "agreed_clauses": list(self._agreed.keys()),
                "episode_score": score,
                "probes_remaining": self._probe_budget - self._probes_used,
            },
        )

    def _calculate_reward(self, action_type, vendor_stance,
                          legal_stance, probe_result) -> float:
        reward = 0.0
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                reward += 0.40
        if action_type == "PROBE" and probe_result:
            reward += 0.10
        if legal_stance == "flagged":
            reward -= 0.20
        if vendor_stance == "walkout":
            reward -= 0.30
        return reward

    def _calculate_final_bonus(self) -> float:
        bonus = 0.0
        agreed_text = " ".join(self._agreed.values()).lower()
        vh = self._deal["vendor_hidden"]
        if vh["value"].lower() in agreed_text:
            bonus += 0.20
        lh = self._deal["legal_hidden"]
        has_redline = any(
            re.search(p, agreed_text, re.IGNORECASE)
            for p in lh["flagged_patterns"]
        )
        if not has_redline:
            bonus += 0.15
        if self._rounds_used < self._round_budget:
            bonus += 0.05
        return bonus

    @property
    def state(self) -> State:
        return self._state