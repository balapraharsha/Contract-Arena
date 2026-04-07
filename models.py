from enum import Enum
from typing import Optional, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class ActionType(str, Enum):
    ACCEPT   = "ACCEPT"
    REJECT   = "REJECT"
    PROPOSE  = "PROPOSE"
    PROBE    = "PROBE"
    ESCALATE = "ESCALATE"


class ContractarenaAction(Action):
    action_type: ActionType = Field(..., description="Type of negotiation action")
    clause_id: Optional[str] = Field(None, description="Target clause ID")
    new_text: Optional[str] = Field(None, description="Proposed clause text (PROPOSE only)")
    reason: Optional[str] = Field(None, description="Reason for rejection (REJECT only)")
    party: Optional[str] = Field(None, description="Party to probe: vendor or legal (PROBE only)")
    question: Optional[str] = Field(None, description="Question to ask (PROBE only)")


class ContractarenaObservation(Observation):
    clause_id: str = Field(default="", description="Current clause being negotiated")
    clause_text: str = Field(default="", description="Current clause text")
    vendor_response: str = Field(default="", description="Vendor's response to last action")
    legal_response: str = Field(default="", description="Legal reviewer's response")
    probe_result: Optional[str] = Field(None, description="Result of PROBE action if used")
    round_number: int = Field(default=0, description="Current round number")
    rounds_remaining: int = Field(default=0, description="Rounds left in episode")
    clauses_agreed: int = Field(default=0, description="Number of clauses agreed so far")
    clauses_total: int = Field(default=0, description="Total clauses in this deal")
    tier: str = Field(default="easy", description="Current difficulty tier")