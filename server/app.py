"""
FastAPI app for ContractArena.
Cycles through all 5 tiers on successive /reset calls:
  easy → medium → hard → expert → marathon → easy → ...
"""
from __future__ import annotations
import itertools

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

try:
    from models import ContractarenaAction, ContractarenaObservation
    from server.contractarena_environment import ContractarenaEnvironment
except ImportError:
    from ..models import ContractarenaAction, ContractarenaObservation
    from .contractarena_environment import ContractarenaEnvironment

_TIER_CYCLE = itertools.cycle(["easy", "medium", "hard", "expert", "marathon"])


def env_factory() -> ContractarenaEnvironment:
    return ContractarenaEnvironment(tier=next(_TIER_CYCLE))


app = create_app(
    env_factory,
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
