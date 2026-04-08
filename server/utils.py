"""
utils.py — shared utilities for ContractArena

Reward clamping is consolidated here rather than duplicated across files.
We clamp to (0.01, 0.99) instead of (0, 1) for two reasons:
  1. OpenEnv platform validators treat exactly-0 as "agent never acted" and
     exactly-1 as "perfect episode" — both cause edge-case logging noise.
  2. The safe_score mapping (0.01 + 0.98 * x) is only well-behaved on the
     open interval; hard boundaries produce identical gradient signal to
     nearby values and confuse reward normalisation in PPO.
"""

from difflib import SequenceMatcher
from typing import List


# ── Reward clamping ──────────────────────────────────────────────────────────

def clamp(value: float) -> float:
    """Hard clamp to open interval (0.01, 0.99)."""
    return round(min(max(float(value), 0.01), 0.99), 4)


def safe_score(value: float) -> float:
    """
    Map an arbitrary float to the strict open interval (0.01, 0.99).
    Used for final episode scores so the platform never sees 0 or 1.
    """
    value = min(max(float(value), 0.001), 0.999)
    result = round(0.01 + 0.98 * value, 4)
    return min(max(result, 0.01), 0.99)


# ── Numerical state-space features ──────────────────────────────────────────

def negotiation_pressure(rounds_used: int, round_budget: int) -> float:
    """Fraction of round budget consumed. Higher → more pressure."""
    if round_budget <= 0:
        return 1.0
    return round(min(rounds_used / round_budget, 1.0), 4)


def per_clause_agreement_rate(clauses_agreed: int, clauses_total: int) -> float:
    """Fraction of clauses successfully closed."""
    if clauses_total <= 0:
        return 0.0
    return round(clauses_agreed / clauses_total, 4)


def vendor_hostility_index(rejection_count: int, total_interactions: int) -> float:
    """Rejection rate — proxy for how hostile the vendor has been."""
    if total_interactions <= 0:
        return 0.0
    return round(min(rejection_count / total_interactions, 1.0), 4)


def legal_risk_score(flagged_count: int, total_patterns: int) -> float:
    """Proportion of legal flagged patterns matched so far."""
    if total_patterns <= 0:
        return 0.0
    return round(min(flagged_count / total_patterns, 1.0), 4)


def probe_efficiency(clauses_agreed: int, probes_used: int) -> float:
    """Information-to-probe ratio — high means probes led to agreements."""
    if probes_used <= 0:
        return 1.0 if clauses_agreed > 0 else 0.0
    return round(min(clauses_agreed / probes_used, 1.0), 4)


def fuzzy_match_score(text: str, target: str) -> float:
    """0–1 similarity between proposed text and vendor's hidden value."""
    if not text or not target:
        return 0.0
    return round(SequenceMatcher(None, text.lower(), target.lower()).ratio(), 4)


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
    Returns a 9-dimensional normalised feature dict suitable for feeding
    directly into an RL policy network alongside the text observation.
    """
    return {
        "negotiation_pressure":    negotiation_pressure(rounds_used, round_budget),
        "clause_agreement_rate":   per_clause_agreement_rate(clauses_agreed, clauses_total),
        "vendor_hostility_index":  vendor_hostility_index(rejection_count, total_interactions),
        "legal_risk_score":        legal_risk_score(flagged_count, total_patterns),
        "probe_efficiency":        probe_efficiency(clauses_agreed, probes_used),
        "rounds_used_norm":        round(min(rounds_used / max(round_budget, 1), 1.0), 4),
        "clauses_remaining_norm":  round(max(clauses_total - clauses_agreed, 0) / max(clauses_total, 1), 4),
        "probes_used_norm":        round(min(probes_used / max(round_budget, 1), 1.0), 4),
        "interaction_count_norm":  round(min(total_interactions / max(round_budget * 2, 1), 1.0), 4),
    }
