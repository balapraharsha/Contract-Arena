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
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContractarenaObservation]:
        """
        Parse server response into StepResult[ContractarenaObservation].
        """
        obs_data = payload.get("observation") or {}

        observation = ContractarenaObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
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