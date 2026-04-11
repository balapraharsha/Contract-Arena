"""
utils.py — shared utilities for ContractArena

Reward clamping is consolidated here. We clamp to (0.01, 0.99) not (0, 1):
  1. OpenEnv validators treat exact 0 as "agent never acted" and exact 1 as
     "perfect episode" — both cause edge-case validation failures.
  2. The safe_score mapping (0.01 + 0.98*x) is only well-behaved on the open
     interval; hard boundaries confuse reward normalisation in PPO/GRPO.

ZOPA/BATNA theory (Raiffa 1982; Fisher & Ury 1981):
  ZOPA  — Zone of Possible Agreement: the range where both parties prefer a
           deal over their outside option. Measured by zopa_utilisation.
  BATNA — Best Alternative To Negotiated Agreement: each party's outside
           option. batna_improvement = deal_score - agent_batna_utility.
  Pareto efficiency — a deal is Pareto-optimal if no party can improve
           without harming the other. Measured as distance from frontier.
"""

import math
from difflib import SequenceMatcher
from typing import List, Dict


# ── Reward clamping ──────────────────────────────────────────────────────────

def clamp(value: float) -> float:
    """Hard clamp to strict open interval (0.01, 0.99)."""
    return round(min(max(float(value), 0.01), 0.99), 4)


def safe_score(value: float) -> float:
    """Map any float to strict open interval (0.01, 0.99). Never returns 0 or 1."""
    value = min(max(float(value), 0.001), 0.999)
    result = round(0.01 + 0.98 * value, 4)
    return min(max(result, 0.01), 0.99)


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def fuzzy_match_score(text: str, target: str) -> float:
    """0–1 similarity between two strings."""
    if not text or not target:
        return 0.0
    return round(SequenceMatcher(None, text.lower(), target.lower()).ratio(), 4)


# ── 9-dim numerical feature vector ───────────────────────────────────────────

def build_numerical_features(
    rounds_used: int,
    round_budget: int,
    clauses_agreed: int,
    clauses_total: int,
    rejection_count: int,
    total_interactions: int,
    flagged_count: int,
    total_patterns: int,
    probes_used: int,
) -> dict:
    """
    9-dimensional normalised feature vector for RL policy networks.
    All values in [0, 1].
    """
    def _div(a, b):
        return round(min(float(a) / max(float(b), 1), 1.0), 4)

    return {
        "negotiation_pressure":   _div(rounds_used, round_budget),
        "clause_agreement_rate":  _div(clauses_agreed, clauses_total),
        "vendor_hostility_index": _div(rejection_count, max(total_interactions, 1)),
        "legal_risk_score":       _div(flagged_count, max(total_patterns, 1)),
        "probe_efficiency":       (round(min(clauses_agreed / probes_used, 1.0), 4)
                                   if probes_used > 0 else (1.0 if clauses_agreed > 0 else 0.0)),
        "rounds_used_norm":       _div(rounds_used, round_budget),
        "clauses_remaining_norm": round(max(clauses_total - clauses_agreed, 0) / max(clauses_total, 1), 4),
        "probes_used_norm":       _div(probes_used, max(round_budget, 1)),
        "interaction_count_norm": _div(total_interactions, max(round_budget * 2, 1)),
    }


# ── ZOPA / BATNA / Pareto theory metrics ─────────────────────────────────────

def compute_pareto_efficiency(
    agent_utility: float,
    vendor_utility: float,
    pareto_frontier: List[Dict],
) -> float:
    """
    How close is the negotiated outcome to the Pareto frontier?
    1.0 = on the frontier (optimal). 0.0 = far from optimal.
    Uses Euclidean distance normalised by sqrt(2).
    """
    if not pareto_frontier:
        return 0.5
    min_dist = min(
        math.sqrt(
            (agent_utility - p.get("agent_utility", 0)) ** 2
            + (vendor_utility - p.get("vendor_utility", 0)) ** 2
        )
        for p in pareto_frontier
    )
    return round(max(0.0, 1.0 - min_dist / math.sqrt(2)), 4)


def compute_zopa_utilisation(
    final_score: float,
    vendor_reservation: float,
    agent_reservation: float,
) -> float:
    """
    What fraction of the ZOPA was captured?
    0 = deal outside ZOPA (bad). 1 = full ZOPA captured (optimal).
    """
    zopa_size = max(vendor_reservation - agent_reservation, 0.01)
    if final_score < agent_reservation:
        return 0.0
    captured = min(final_score - agent_reservation, zopa_size)
    return round(captured / zopa_size, 4)


def compute_batna_improvement(
    final_score: float,
    agent_batna_utility: float,
) -> float:
    """
    How much better is the deal vs the agent's BATNA?
    Positive = deal is rational. Negative = should have walked away.
    """
    return round(final_score - agent_batna_utility, 4)


def compute_counterfactual_optimal(
    clauses_total: int,
    round_budget: int,
    probe_budget: int,
) -> float:
    """
    Estimate what a perfect-information agent would score.
    Used to compute efficiency_ratio = actual / optimal.
    """
    optimal = clauses_total * 0.40 + 0.40
    max_possible = max(clauses_total * 0.40 + 0.40, 0.01)
    return safe_score(optimal / max_possible)


def compute_efficiency_ratio(actual: float, optimal: float) -> float:
    """
    efficiency_ratio = actual / optimal.
    Measures the cost of information asymmetry.
    1.0 = perfect play. 0.5 = agent got half the available value.
    """
    if optimal <= 0:
        return 0.0
    return round(min(actual / optimal, 1.0), 4)
