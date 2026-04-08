"""
contractarena_environment.py — Core RL environment for ContractArena.

Key design decisions:
  - Reward clamping is centralised in utils.py (see there for rationale on (0.01, 0.99) bounds)
  - Opponents are stochastic per-episode to prevent policy memorisation
  - Observation includes a 9-dim numerical feature vector for RL policy networks
  - Expert tier adds a silent third stakeholder (ComplianceOfficer) with geometric mean scoring
  - Partial reward shaping gives dense signal without hiding the terminal bonus structure
"""

import json
import re
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics.base import Rubric

try:
    from ..models import ContractarenaAction, ContractarenaObservation
    from .opponents import VendorAgent, LegalReviewer
    from .utils import clamp, safe_score, build_numerical_features, fuzzy_match_score
except ImportError:
    from models import ContractarenaAction, ContractarenaObservation
    from server.opponents import VendorAgent, LegalReviewer
    from server.utils import clamp, safe_score, build_numerical_features, fuzzy_match_score

DEALS_DIR = Path(__file__).parent / "deals"


def load_deal(tier: str) -> Dict[str, Any]:
    path = DEALS_DIR / f"{tier}.json"
    if not path.exists():
        raise FileNotFoundError(f"Deal file not found for tier '{tier}': {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Compliance Officer (expert tier silent stakeholder) ───────────────────────

class ComplianceOfficer:
    """
    Silent third stakeholder present only in the expert tier.
    The agent receives no direct compliance_response — it must deduce
    compliance stance from probe_result or by observing flag signals.
    """

    def __init__(self, config: dict):
        self._redline = config["redline"]
        self._required_keyword = config["required_keyword"]
        self._flagged_patterns = config["flagged_patterns"]
        self._stakeholder = config.get("stakeholder", "Compliance Officer")
        self._flagged_count = 0

    def review(self, action_type: str, new_text: str = "") -> tuple:
        action_type = (action_type or "").upper()
        if action_type == "PROBE":
            return (
                f"{self._stakeholder} requires '{self._required_keyword}' "
                f"in any clause touching {self._redline}.",
                "noted",
            )
        if action_type in ("ACCEPT", "PROPOSE"):
            text = new_text or ""
            for pat in self._flagged_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    self._flagged_count += 1
                    return (
                        f"{self._stakeholder} blocks: {self._redline} violation detected.",
                        "blocked",
                    )
            # Check for required keyword
            if self._required_keyword.replace("_", " ") in text.lower() or \
               self._required_keyword in text.lower():
                return f"{self._stakeholder} approves.", "approved"
            # Neutral — neither flagged nor explicitly approved
            return f"{self._stakeholder} is reviewing.", "reviewing"
        return f"{self._stakeholder} is monitoring.", "reviewing"

    def reset(self):
        self._flagged_count = 0

    @property
    def flagged_count(self) -> int:
        return self._flagged_count


# ── Rubric ─────────────────────────────────────────────────────────────────────

class ContractArenaRubric(Rubric):
    def __init__(self, env: "ContractarenaEnvironment"):
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        rewards = self._env._episode_rewards
        if not rewards:
            return 0.01
        raw = sum(rewards)
        max_possible = max(len(self._env._clauses) * 0.40 + 0.40, 0.01)
        normalized = raw / max_possible
        score = safe_score(normalized)
        # Expert tier: geometric mean penalty if compliance score is poor
        if self._env._tier == "expert":
            compliance_score = self._env._compliance_score()
            score = safe_score((score * compliance_score) ** 0.5)
        return score

    def reset(self) -> None:
        pass


# ── Environment ────────────────────────────────────────────────────────────────

class ContractarenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, tier: str = "easy"):
        self._tier = tier
        self._deal = load_deal(tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._compliance_officer = None
        self._init_opponents()
        self._reset_episode()
        self.rubric = ContractArenaRubric(self)

    def _init_opponents(self) -> None:
        vendor_hidden = self._deal["vendor_hidden"]
        legal_hidden = self._deal["legal_hidden"]
        self._vendor = VendorAgent(
            hidden_priority=vendor_hidden["priority"],
            hidden_value=vendor_hidden["value"],
            walkout_threshold=vendor_hidden.get("walkout_threshold", 3),
        )
        self._legal = LegalReviewer(
            hidden_redline=legal_hidden["redline"],
            hidden_value=legal_hidden["value"],
            flagged_patterns=legal_hidden["flagged_patterns"],
        )
        # Expert tier: add silent compliance officer
        if self._tier == "expert" and "compliance_hidden" in self._deal:
            self._compliance_officer = ComplianceOfficer(self._deal["compliance_hidden"])

    def _reset_episode(self) -> None:
        self._clauses = list(self._deal["clauses"])
        self._clause_index = 0
        self._agreed: Dict[str, str] = {}
        self._round_budget = int(self._deal["round_budget"])
        self._probe_budget = int(self._deal.get("probe_budget", 999))
        self._probes_used = 0
        self._rounds_used = 0
        self._total_interactions = 0
        self._episode_rewards: list = []
        self._known_probe_results: set = set()  # for redundant-probe detection
        self._prev_vendor_stance = "open"
        self._vendor.reset()
        self._legal.reset()
        if self._compliance_officer:
            self._compliance_officer.reset()

    def _compliance_score(self) -> float:
        """Geometric-mean penalty factor for expert tier."""
        if not self._compliance_officer:
            return 1.0
        total = len(self._clauses)
        flagged = self._compliance_officer.flagged_count
        if total <= 0:
            return 1.0
        return max(0.01, 1.0 - (flagged / total))

    def reset(self) -> ContractarenaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_episode()
        if self.rubric is not None:
            self.rubric.reset()
        clause = self._clauses[0]
        features = build_numerical_features(0, self._round_budget, 0, len(self._clauses), 0, 0, 0, 0, 0)
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
            reward=0.01,
            metadata={
                "vendor_stance": "open",
                "legal_stance": "approved",
                "agreed_clauses": [],
                "episode_score": 0.01,
                "probes_remaining": self._probe_budget,
                "numerical_features": features,
            },
        )

    def step(self, action: ContractarenaAction) -> ContractarenaObservation:
        self._state.step_count += 1
        self._rounds_used += 1
        self._total_interactions += 1

        clause = self._clauses[self._clause_index]
        clause_id = clause["id"]
        new_text = action.new_text or clause["text"]
        action_type = action.action_type.value

        vendor_resp, vendor_stance = self._vendor.respond(action_type, clause_id, new_text)
        legal_resp, legal_stance = self._legal.review(action_type, clause_id, new_text)

        # Expert tier: compliance officer also reviews
        compliance_resp, compliance_stance = None, "reviewing"
        if self._compliance_officer and action_type in ("ACCEPT", "PROPOSE", "PROBE"):
            compliance_resp, compliance_stance = self._compliance_officer.review(action_type, new_text)

        probe_result = None
        if action_type == "PROBE":
            if self._probes_used >= self._probe_budget:
                probe_result = "Probe budget exhausted — no more information available."
                vendor_resp = "No further information."
                legal_resp = "No further information."
            else:
                self._probes_used += 1
                party = (action.party or "vendor").lower()
                if party == "compliance" and compliance_resp:
                    probe_result = compliance_resp
                elif party == "vendor":
                    probe_result = vendor_resp
                else:
                    probe_result = legal_resp

        reward = self._calculate_reward(
            action_type=action_type,
            vendor_stance=vendor_stance,
            legal_stance=legal_stance,
            compliance_stance=compliance_stance,
            probe_result=probe_result,
            new_text=new_text,
        )

        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                self._agreed[clause_id] = new_text
                self._clause_index = min(self._clause_index + 1, len(self._clauses) - 1)

        self._prev_vendor_stance = vendor_stance
        self._episode_rewards.append(clamp(reward))

        all_agreed = len(self._agreed) == len(self._clauses)
        walkout = vendor_stance == "walkout"
        out_of_rounds = self._rounds_used >= self._round_budget
        done = all_agreed or walkout or out_of_rounds

        if done:
            bonus = self._calculate_final_bonus()
            reward = clamp(reward + bonus)
            self._episode_rewards[-1] = reward

        reward = clamp(reward)

        raw_total = sum(self._episode_rewards)
        max_possible = max(len(self._clauses) * 0.40 + 0.40, 0.01)
        normalized = raw_total / max_possible
        score = safe_score(normalized)
        if self._tier == "expert":
            cs = self._compliance_score()
            score = safe_score((score * cs) ** 0.5)

        if not done and self._clause_index < len(self._clauses):
            next_clause = self._clauses[self._clause_index]
        else:
            next_clause = clause

        features = build_numerical_features(
            rounds_used=self._rounds_used,
            round_budget=self._round_budget,
            clauses_agreed=len(self._agreed),
            clauses_total=len(self._clauses),
            rejection_count=self._vendor.rejection_count,
            total_interactions=self._total_interactions,
            flagged_count=self._legal.flagged_count,
            total_patterns=self._legal.total_patterns,
            probes_used=self._probes_used,
        )

        return ContractarenaObservation(
            clause_id=next_clause["id"],
            clause_text=next_clause["text"],
            vendor_response=vendor_resp,
            legal_response=legal_resp,
            probe_result=probe_result,
            round_number=self._rounds_used,
            rounds_remaining=max(self._round_budget - self._rounds_used, 0),
            clauses_agreed=len(self._agreed),
            clauses_total=len(self._clauses),
            tier=self._tier,
            done=done,
            reward=reward,
            metadata={
                "vendor_stance": vendor_stance,
                "legal_stance": legal_stance,
                "agreed_clauses": list(self._agreed.keys()),
                "episode_score": score,
                "probes_remaining": max(self._probe_budget - self._probes_used, 0),
                "numerical_features": features,
                "compliance_stance": compliance_stance,
            },
        )

    def _calculate_reward(
        self,
        action_type: str,
        vendor_stance: str,
        legal_stance: str,
        compliance_stance: str,
        probe_result: str | None,
        new_text: str = "",
    ) -> float:
        reward = 0.01

        # ── Primary agreement signal ─────────────────────────────────────────
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                reward += 0.40

        # ── Probe signal ─────────────────────────────────────────────────────
        if action_type == "PROBE" and probe_result:
            if probe_result in self._known_probe_results:
                # Redundant probe — the agent already had this information
                reward -= 0.05
            else:
                self._known_probe_results.add(probe_result)
                reward += 0.10

        # ── Partial reward shaping (dense intermediate signals) ───────────────
        # +0.05 when vendor stance moves from firm → open (progress without full agreement)
        if self._prev_vendor_stance == "firm" and vendor_stance == "open":
            reward += 0.05

        # +0.08 when proposed text contains the vendor's hidden value (fuzzy)
        if action_type == "PROPOSE" and new_text:
            similarity = fuzzy_match_score(new_text, self._vendor.hidden_value)
            if similarity > 0.6 and vendor_stance != "open":
                reward += 0.08 * similarity  # scaled by how close the match is

        # ── Penalties ────────────────────────────────────────────────────────
        if legal_stance == "flagged":
            reward -= 0.20
        if vendor_stance == "walkout":
            reward -= 0.30
        # Expert tier: compliance block is an additional penalty
        if compliance_stance == "blocked":
            reward -= 0.15

        return clamp(reward)

    def _calculate_final_bonus(self) -> float:
        bonus = 0.0
        agreed_text = " ".join(self._agreed.values()).lower()

        vendor_hidden = self._deal["vendor_hidden"]
        if vendor_hidden["value"].lower() in agreed_text:
            bonus += 0.20

        legal_hidden = self._deal["legal_hidden"]
        has_redline = any(
            re.search(pattern, agreed_text, re.IGNORECASE)
            for pattern in legal_hidden["flagged_patterns"]
        )
        if not has_redline:
            bonus += 0.15

        if self._rounds_used < self._round_budget:
            bonus += 0.05

        # Expert tier: bonus for satisfying compliance officer
        if self._compliance_officer and "compliance_hidden" in self._deal:
            kw = self._deal["compliance_hidden"]["required_keyword"]
            if kw.replace("_", " ") in agreed_text or kw in agreed_text:
                bonus += 0.10

        return clamp(bonus)

    @property
    def state(self) -> State:
        return self._state
