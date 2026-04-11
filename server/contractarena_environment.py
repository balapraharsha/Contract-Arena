"""
contractarena_environment.py — Core RL environment for ContractArena.

Design decisions:
  - Reward clamping centralised in utils.py
  - Stochastic opponents per episode (walkout threshold ±1, surprise legal patterns)
  - 9-dim numerical feature vector for RL policy networks
  - ZOPA/BATNA/Pareto theory: efficiency_ratio, zopa_utilisation, pareto_efficiency
  - Counterfactual optimal score logged every episode
  - Expert tier: silent ComplianceOfficer, geometric mean scoring
  - Hard tier: vendor-legal coalition dynamics
  - Marathon tier: 3 back-to-back deals with knowledge transfer
  - Dense partial reward shaping for strong gradient signal
"""

import json
import re
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics.base import Rubric

try:
    from ..models import ContractarenaAction, ContractarenaObservation
    from .opponents import VendorAgent, LegalReviewer
    from .utils import (
        clamp, safe_score, build_numerical_features, fuzzy_match_score,
        compute_pareto_efficiency, compute_zopa_utilisation,
        compute_batna_improvement, compute_counterfactual_optimal,
        compute_efficiency_ratio,
    )
except ImportError:
    from models import ContractarenaAction, ContractarenaObservation
    from server.opponents import VendorAgent, LegalReviewer
    from server.utils import (
        clamp, safe_score, build_numerical_features, fuzzy_match_score,
        compute_pareto_efficiency, compute_zopa_utilisation,
        compute_batna_improvement, compute_counterfactual_optimal,
        compute_efficiency_ratio,
    )

DEALS_DIR = Path(__file__).parent / "deals"


def load_deal(tier: str) -> Dict[str, Any]:
    path = DEALS_DIR / f"{tier}.json"
    if not path.exists():
        raise FileNotFoundError(f"Deal file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Compliance Officer ────────────────────────────────────────────────────────

class ComplianceOfficer:
    """
    Silent third stakeholder (expert tier only).
    Agent must probe party='compliance' to learn requirements.
    Final score uses geometric mean: score * compliance_score.
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
                    return f"{self._stakeholder} blocks: {self._redline} violation.", "blocked"
            kw = self._required_keyword
            if kw.replace("_", " ") in text.lower() or kw in text.lower():
                return f"{self._stakeholder} approves.", "approved"
            return f"{self._stakeholder} is reviewing.", "reviewing"
        return f"{self._stakeholder} is monitoring.", "reviewing"

    def reset(self):
        self._flagged_count = 0

    @property
    def flagged_count(self) -> int:
        return self._flagged_count


# ── Rubric ────────────────────────────────────────────────────────────────────

class ContractArenaRubric(Rubric):
    def __init__(self, env: "ContractarenaEnvironment"):
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        rewards = self._env._episode_rewards
        if not rewards:
            return 0.01
        raw = sum(rewards)
        max_possible = max(self._env._total_clauses_count * 0.40 + 0.40, 0.01)
        score = safe_score(raw / max_possible)
        if self._env._tier == "expert":
            cs = self._env._compliance_score()
            score = safe_score((score * cs) ** 0.5)
        return score

    def reset(self) -> None:
        pass


# ── Environment ───────────────────────────────────────────────────────────────

class ContractarenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, tier: str = "easy"):
        self._tier = tier
        self._deal = load_deal(tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._compliance_officer: Optional[ComplianceOfficer] = None

        # Marathon state
        self._marathon_deals: List[Dict] = []
        self._marathon_index: int = 0
        self._marathon_knowledge: Dict[str, str] = {}
        self._total_clauses_count: int = 0

        self._init_opponents()
        self._reset_episode()
        self.rubric = ContractArenaRubric(self)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_opponents(self) -> None:
        if self._tier == "marathon":
            self._marathon_deals = self._deal.get("deals", [])
            self._total_clauses_count = sum(len(d["clauses"]) for d in self._marathon_deals)
            self._load_marathon_deal(0)
            return

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
        self._clauses = list(self._deal["clauses"])
        self._total_clauses_count = len(self._clauses)

        if self._tier == "expert" and "compliance_hidden" in self._deal:
            self._compliance_officer = ComplianceOfficer(self._deal["compliance_hidden"])

    def _load_marathon_deal(self, index: int) -> None:
        deals = self._marathon_deals
        if index >= len(deals):
            return
        d = deals[index]
        self._vendor = VendorAgent(
            hidden_priority=d["vendor_hidden"]["priority"],
            hidden_value=d["vendor_hidden"]["value"],
            walkout_threshold=d["vendor_hidden"].get("walkout_threshold", 3),
        )
        self._legal = LegalReviewer(
            hidden_redline=d["legal_hidden"]["redline"],
            hidden_value=d["legal_hidden"]["value"],
            flagged_patterns=d["legal_hidden"]["flagged_patterns"],
        )
        self._clauses = list(d["clauses"])

    def _reset_episode(self) -> None:
        if self._tier == "marathon":
            self._marathon_index = 0
            self._marathon_knowledge = {}
            self._load_marathon_deal(0)

        self._clause_index = 0
        self._agreed: Dict[str, str] = {}
        self._all_agreed_text: List[str] = []
        self._round_budget = int(self._deal["round_budget"])
        self._probe_budget = int(self._deal.get("probe_budget", 999))
        self._probes_used = 0
        self._rounds_used = 0
        self._total_interactions = 0
        self._episode_rewards: List[float] = []
        self._known_probe_results: set = set()
        self._prev_vendor_stance = "open"
        self._vendor.reset()
        self._legal.reset()
        if self._compliance_officer:
            self._compliance_officer.reset()

    def _compliance_score(self) -> float:
        if not self._compliance_officer:
            return 1.0
        flagged = self._compliance_officer.flagged_count
        return max(0.01, 1.0 - flagged / max(len(self._clauses), 1))

    # ── Metadata builder ──────────────────────────────────────────────────────

    def _build_metadata(
        self,
        vendor_stance: str,
        legal_stance: str,
        compliance_stance: str,
        score: float,
        done: bool = False,
    ) -> dict:
        zopa = self._deal.get("zopa", {})
        pareto_eff = 0.0
        zopa_util = 0.0
        batna_imp = 0.0

        if done and zopa.get("exists"):
            vendor_util = min(len(self._agreed) / max(len(self._clauses), 1), 1.0)
            pareto_eff = compute_pareto_efficiency(
                score, vendor_util, zopa.get("pareto_frontier", [])
            )
            zopa_util = compute_zopa_utilisation(
                score,
                zopa.get("vendor_reservation_price", 0.5),
                zopa.get("agent_reservation_price", 0.4),
            )
            batna_imp = compute_batna_improvement(
                score,
                self._deal.get("vendor_hidden", {}).get("batna_utility", 0.3),
            )

        optimal = compute_counterfactual_optimal(
            len(self._clauses), self._round_budget, self._probe_budget
        )
        eff_ratio = compute_efficiency_ratio(score, optimal)

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

        return {
            "vendor_stance":          vendor_stance,
            "legal_stance":           legal_stance,
            "compliance_stance":      compliance_stance,
            "agreed_clauses":         list(self._agreed.keys()),
            "episode_score":          score,
            "probes_remaining":       max(self._probe_budget - self._probes_used, 0),
            "numerical_features":     features,
            "counterfactual_optimal": optimal,
            "efficiency_ratio":       eff_ratio,
            "zopa_utilisation":       zopa_util,
            "pareto_efficiency":      pareto_eff,
            "batna_improvement":      batna_imp,
            "marathon_deal":          self._marathon_index + 1 if self._tier == "marathon" else None,
            "marathon_knowledge":     dict(self._marathon_knowledge) if self._tier == "marathon" else {},
        }

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> ContractarenaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_episode()
        if self.rubric is not None:
            self.rubric.reset()
        clause = self._clauses[0]
        meta = self._build_metadata("open", "approved", "reviewing", 0.01)
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
            metadata=meta,
        )

    # ── step ──────────────────────────────────────────────────────────────────

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

        # Coalition check (hard tier)
        coalition = self._deal.get("coalition", {})
        if coalition.get("active") and action_type in ("ACCEPT", "PROPOSE"):
            trigger = coalition.get("trigger_pattern", "")
            if trigger and re.search(trigger, new_text, re.IGNORECASE):
                if vendor_stance == "open":
                    vendor_stance = "firm"
                    vendor_resp += f" [Coalition: vendor aligns with legal on '{trigger}']"

        # Expert compliance review
        compliance_resp, compliance_stance = None, "reviewing"
        if self._compliance_officer and action_type in ("ACCEPT", "PROPOSE", "PROBE"):
            compliance_resp, compliance_stance = self._compliance_officer.review(
                action_type, new_text
            )

        # Probe handling
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
                    if self._tier == "marathon":
                        self._marathon_knowledge[
                            f"deal{self._marathon_index+1}_vendor"
                        ] = probe_result
                else:
                    probe_result = legal_resp

        reward = self._calculate_reward(
            action_type=action_type,
            vendor_stance=vendor_stance,
            legal_stance=legal_stance,
            compliance_stance=compliance_stance,
            probe_result=probe_result,
            new_text=new_text,
            coalition_active=coalition.get("active", False),
        )

        # Agreement tracking
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                self._agreed[clause_id] = new_text
                self._all_agreed_text.append(new_text)
                self._clause_index = min(self._clause_index + 1, len(self._clauses) - 1)

        self._prev_vendor_stance = vendor_stance
        self._episode_rewards.append(clamp(reward))

        all_agreed = len(self._agreed) == len(self._clauses)
        walkout = vendor_stance == "walkout"
        out_of_rounds = self._rounds_used >= self._round_budget

        # Marathon: advance to next sub-deal when current completes
        if all_agreed and self._tier == "marathon":
            self._marathon_index += 1
            if self._marathon_index < len(self._marathon_deals):
                self._load_marathon_deal(self._marathon_index)
                self._vendor.reset()
                self._legal.reset()
                self._agreed = {}
                self._clause_index = 0
                all_agreed = False

        done = all_agreed or walkout or out_of_rounds

        if done:
            bonus = self._calculate_final_bonus()
            reward = clamp(reward + bonus)
            self._episode_rewards[-1] = reward

        reward = clamp(reward)

        # Score computation
        raw_total = sum(self._episode_rewards)
        max_possible = max(self._total_clauses_count * 0.40 + 0.40, 0.01)
        score = safe_score(raw_total / max_possible)
        if self._tier == "expert":
            cs = self._compliance_score()
            score = safe_score((score * cs) ** 0.5)

        next_clause = (
            self._clauses[self._clause_index]
            if not done and self._clause_index < len(self._clauses)
            else clause
        )

        meta = self._build_metadata(vendor_stance, legal_stance, compliance_stance, score, done)

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
            metadata=meta,
        )

    # ── reward ────────────────────────────────────────────────────────────────

    def _calculate_reward(
        self,
        action_type: str,
        vendor_stance: str,
        legal_stance: str,
        compliance_stance: str,
        probe_result: Optional[str],
        new_text: str = "",
        coalition_active: bool = False,
    ) -> float:
        reward = 0.01

        # Primary agreement
        if vendor_stance == "open" and legal_stance == "approved":
            if action_type in ("ACCEPT", "PROPOSE"):
                reward += 0.40

        # Probe signal
        if action_type == "PROBE" and probe_result:
            if probe_result in self._known_probe_results:
                reward -= 0.05  # redundant probe penalty
            else:
                self._known_probe_results.add(probe_result)
                reward += 0.10

        # Partial shaping: vendor stance firm→open
        if self._prev_vendor_stance == "firm" and vendor_stance == "open":
            reward += 0.05

        # Partial shaping: proposal fuzzy-matches vendor hidden value
        if action_type == "PROPOSE" and new_text:
            sim = fuzzy_match_score(new_text, self._vendor.hidden_value)
            if sim > 0.6 and vendor_stance != "open":
                reward += 0.08 * sim

        # Marathon: bonus for using transferred knowledge
        if self._tier == "marathon" and action_type == "PROPOSE" and self._marathon_index > 0:
            for known in self._marathon_knowledge.values():
                if known and known.lower() in new_text.lower():
                    reward += 0.05
                    break

        # Penalties
        if legal_stance == "flagged":
            reward -= 0.20
        if vendor_stance == "walkout":
            reward -= 0.30
        if compliance_stance == "blocked":
            reward -= 0.15
        if coalition_active and vendor_stance == "firm" and legal_stance == "flagged":
            reward -= 0.10  # extra coalition coordination penalty

        return clamp(reward)

    def _calculate_final_bonus(self) -> float:
        bonus = 0.0
        agreed_text = " ".join(self._all_agreed_text).lower()

        if self._deal["vendor_hidden"]["value"].lower() in agreed_text:
            bonus += 0.20

        if not any(
            re.search(p, agreed_text, re.IGNORECASE)
            for p in self._deal["legal_hidden"]["flagged_patterns"]
        ):
            bonus += 0.15

        if self._rounds_used < self._round_budget:
            bonus += 0.05

        if self._compliance_officer and "compliance_hidden" in self._deal:
            kw = self._deal["compliance_hidden"]["required_keyword"]
            if kw.replace("_", " ") in agreed_text or kw in agreed_text:
                bonus += 0.10

        if self._tier == "marathon" and self._marathon_knowledge:
            if self._marathon_index >= len(self._marathon_deals) - 1:
                bonus += 0.10

        return clamp(bonus)

    @property
    def state(self) -> State:
        return self._state
