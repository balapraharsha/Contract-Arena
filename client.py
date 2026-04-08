# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contractarena Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ContractarenaAction, ContractarenaObservation


def _safe_reward(value: Any) -> float:
    """Ensure reward is strictly between 0 and 1."""
    try:
        v = float(value) if value is not None else 0.01
    except (TypeError, ValueError):
        v = 0.01
    v = min(max(v, 0.0), 1.0)
    return round(0.01 + 0.98 * v, 4)


class ContractarenaEnv(
    EnvClient[ContractarenaAction, ContractarenaObservation, State]
):
    """
    Client for the Contractarena Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: ContractarenaAction) -> Dict[str, Any]:
        """
        Convert ContractarenaAction to JSON payload for step message.
        """
        payload: Dict[str, Any] = {
            "action_type": action.action_type.value,
        }
        if action.clause_id is not None:
            payload["clause_id"] = action.clause_id
        if action.new_text is not None:
            payload["new_text"] = action.new_text
        if action.reason is not None:
            payload["reason"] = action.reason
        if action.party is not None:
            payload["party"] = action.party
        if action.question is not None:
            payload["question"] = action.question
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContractarenaObservation]:
        """
        Parse server response into StepResult[ContractarenaObservation].
        """
        obs_data = payload.get("observation") or {}
        raw_reward = payload.get("reward", 0.01)
        safe_rew = _safe_reward(raw_reward)

        observation = ContractarenaObservation(
            clause_id=obs_data.get("clause_id", ""),
            clause_text=obs_data.get("clause_text", ""),
            vendor_response=obs_data.get("vendor_response", ""),
            legal_response=obs_data.get("legal_response", ""),
            probe_result=obs_data.get("probe_result"),
            round_number=obs_data.get("round_number", 0),
            rounds_remaining=obs_data.get("rounds_remaining", 0),
            clauses_agreed=obs_data.get("clauses_agreed", 0),
            clauses_total=obs_data.get("clauses_total", 0),
            tier=obs_data.get("tier", "easy"),
            done=payload.get("done", False),
            reward=safe_rew,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=safe_rew,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )