# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Contractarena environment.

Exposes the environment through OpenEnv-compatible HTTP/WebSocket endpoints.
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies and ensure the OpenEnv package is available."
    ) from e

try:
    from models import ContractarenaAction, ContractarenaObservation
    from server.contractarena_environment import ContractarenaEnvironment
except ImportError:
    from ..models import ContractarenaAction, ContractarenaObservation
    from .contractarena_environment import ContractarenaEnvironment


app = create_app(
    ContractarenaEnvironment,
    ContractarenaAction,
    ContractarenaObservation,
    env_name="contractarena",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()