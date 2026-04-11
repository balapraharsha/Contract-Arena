"""
opponents.py — VendorAgent and LegalReviewer with per-episode stochasticity.

Each episode randomises:
  - Vendor walkout threshold (2–4) — agent cannot memorise a fixed budget
  - A secondary concern clause for vendor (adds noise to responses)
  - Legal occasionally adds a surprise redline pattern (~15% chance)

This prevents a trained policy from collapsing to deterministic lookup
and forces genuine generalisation across episodes.
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple


_SURPRISE_LEGAL_PATTERNS = [
    r"in perpetuity",
    r"irrevocable",
    r"sole discretion",
    r"without recourse",
    r"as-is",
]


@dataclass
class VendorAgent:
    hidden_priority: str
    hidden_value: str
    walkout_threshold: int = 3
    _rejection_count: int = field(default=0, repr=False)
    _probe_used: bool = field(default=False, repr=False)
    _effective_walkout: int = field(default=0, repr=False)
    _random_care_clause: str = field(default="", repr=False)

    def reset(self) -> None:
        self._rejection_count = 0
        self._probe_used = False
        self._effective_walkout = max(2, min(4, self.walkout_threshold + random.randint(-1, 1)))
        self._random_care_clause = random.choice(
            ["payment schedule", "support SLA", "audit rights", ""]
        )

    def respond(self, action_type: str, clause_id: str = "", new_text: str = "") -> Tuple[str, str]:
        action_type = (action_type or "").upper()
        if self._effective_walkout == 0:
            self._effective_walkout = self.walkout_threshold

        if action_type == "ACCEPT":
            return "Vendor accepts this clause.", "open"

        if action_type == "PROBE":
            if not self._probe_used:
                self._probe_used = True
                extra = (f" We also want clarity on {self._random_care_clause}."
                         if self._random_care_clause else "")
                return f"We care strongly about {self.hidden_priority}.{extra}", "open"
            else:
                return (f"To be clear: we need {self.hidden_value} "
                        f"for {self.hidden_priority}.", "open")

        if action_type == "ESCALATE":
            return (f"Under escalation: we require {self.hidden_value} for "
                    f"{self.hidden_priority}. This is non-negotiable.", "firm")

        if action_type == "PROPOSE":
            if self.hidden_value.lower() in (new_text or "").lower():
                return "Vendor agrees — the proposed terms are acceptable.", "open"
            self._rejection_count += 1
            if self._rejection_count >= self._effective_walkout:
                return "Vendor is walking out of negotiations.", "walkout"
            remaining = self._effective_walkout - self._rejection_count
            return (f"Vendor is firm. Our key requirement is {self.hidden_priority}. "
                    f"({remaining} rejection(s) before walkout)", "firm")

        if action_type == "REJECT":
            return "Vendor notes the rejection and awaits a counter-proposal.", "firm"

        return "Vendor is waiting.", "open"

    @property
    def rejection_count(self) -> int:
        return self._rejection_count

    @property
    def effective_walkout_threshold(self) -> int:
        return self._effective_walkout


@dataclass
class LegalReviewer:
    hidden_redline: str
    hidden_value: str
    flagged_patterns: list
    _probe_used: bool = field(default=False, repr=False)
    _surprise_pattern: str = field(default="", repr=False)
    _flagged_count: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._probe_used = False
        self._flagged_count = 0
        self._surprise_pattern = (
            random.choice(_SURPRISE_LEGAL_PATTERNS)
            if random.random() < 0.15 else ""
        )

    def _all_patterns(self) -> List[str]:
        patterns = list(self.flagged_patterns)
        if self._surprise_pattern:
            patterns.append(self._surprise_pattern)
        return patterns

    def review(self, action_type: str, clause_id: str = "", new_text: str = "") -> Tuple[str, str]:
        action_type = (action_type or "").upper()

        if action_type == "PROBE":
            self._probe_used = True
            extra = (f" (Also watching for '{self._surprise_pattern}' this session.)"
                     if self._surprise_pattern else "")
            return f"Legal has concerns about {self.hidden_redline}.{extra}", "approved"

        if action_type == "ESCALATE":
            return (f"Legal redline confirmed: {self.hidden_value} is mandatory "
                    f"for {self.hidden_redline}.", "flagged")

        if action_type in ("ACCEPT", "PROPOSE"):
            text = new_text or ""
            for pattern in self._all_patterns():
                if re.search(pattern, text, re.IGNORECASE):
                    self._flagged_count += 1
                    note = (f" (Triggered surprise redline: '{pattern}')"
                            if pattern == self._surprise_pattern else "")
                    return (f"Legal flags this clause: {self.hidden_redline} detected. "
                            f"Requires {self.hidden_value}.{note}", "flagged")
            return "Legal approves this clause.", "approved"

        if action_type == "REJECT":
            return "Legal notes the rejection.", "approved"

        return "Legal is reviewing.", "approved"

    @property
    def flagged_count(self) -> int:
        return self._flagged_count

    @property
    def total_patterns(self) -> int:
        return len(self._all_patterns())
