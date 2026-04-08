import re
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VendorAgent:
    hidden_priority: str
    hidden_value: str
    walkout_threshold: int = 3
    _rejection_count: int = field(default=0, repr=False)
    _probe_used: bool = field(default=False, repr=False)

    def respond(self, action_type: str, clause_id: str = "", new_text: str = "") -> Tuple[str, str]:
        """Returns (response_text, new_stance)."""

        action_type = (action_type or "").upper()

        if action_type == "ACCEPT":
            return "Vendor accepts this clause.", "open"

        if action_type == "PROBE":
            if not self._probe_used:
                self._probe_used = True
                return f"We care strongly about {self.hidden_priority}.", "open"
            else:
                return (
                    f"To be clear: we need {self.hidden_value} "
                    f"for {self.hidden_priority}.",
                    "open"
                )

        if action_type == "ESCALATE":
            return (
                f"Under escalation: we require {self.hidden_value} for {self.hidden_priority}. "
                f"This is non-negotiable.",
                "firm",
            )

        if action_type == "PROPOSE":
            if self.hidden_value.lower() in (new_text or "").lower():
                return "Vendor agrees — the proposed terms are acceptable.", "open"

            self._rejection_count += 1
            if self._rejection_count >= self.walkout_threshold:
                return "Vendor is walking out of negotiations.", "walkout"

            return f"Vendor is firm. Our key requirement is {self.hidden_priority}.", "firm"

        if action_type == "REJECT":
            return "Vendor notes the rejection and awaits a counter-proposal.", "firm"

        return "Vendor is waiting.", "open"

    def reset(self) -> None:
        self._rejection_count = 0
        self._probe_used = False


@dataclass
class LegalReviewer:
    hidden_redline: str
    hidden_value: str
    flagged_patterns: list
    _probe_used: bool = field(default=False, repr=False)

    def review(self, action_type: str, clause_id: str = "", new_text: str = "") -> Tuple[str, str]:
        """Returns (response_text, new_stance)."""

        action_type = (action_type or "").upper()

        if action_type == "PROBE":
            self._probe_used = True
            if self._probe_used:
                return f"Legal has concerns about {self.hidden_redline}.", "approved"

        if action_type == "ESCALATE":
            return (
                f"Legal redline confirmed: {self.hidden_value} is mandatory for {self.hidden_redline}.",
                "flagged",
            )

        if action_type in ("ACCEPT", "PROPOSE"):
            text_to_check = new_text or ""
            for pattern in self.flagged_patterns:
                if re.search(pattern, text_to_check, re.IGNORECASE):
                    return (
                        f"Legal flags this clause: {self.hidden_redline} detected. Requires {self.hidden_value}.",
                        "flagged",
                    )
            return "Legal approves this clause.", "approved"

        if action_type == "REJECT":
            return "Legal notes the rejection.", "approved"

        return "Legal is reviewing.", "approved"

    def reset(self) -> None:
        self._probe_used = False