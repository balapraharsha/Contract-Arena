"""
trajectory_collector.py — Episode trajectory logging and GRPO data formatter.

Logs every episode to JSONL and exposes a toy fine-tuning data formatter
compatible with GRPO (Group Relative Policy Optimisation) pipelines.

Usage:
    # Collect trajectories while running inference
    from trajectory_collector import TrajectoryCollector

    collector = TrajectoryCollector("trajectories.jsonl")
    collector.start_episode(tier="easy", episode_id="ep_001")
    collector.log_step(action, observation, reward)
    collector.end_episode(final_score=0.82)

    # Export GRPO-formatted dataset
    collector.export_grpo("grpo_dataset.jsonl")

GRPO Format (compatible with trl.GRPOTrainer):
    Each line is a JSON object with:
      - prompt: str          (the observation as a text prompt)
      - completions: list    (list of action strings tried in this state)
      - rewards: list        (corresponding rewards for each completion)
      - metadata: dict       (tier, episode_id, step, numerical_features)

    The group structure (multiple completions per prompt) is populated by
    running multiple rollouts from the same state — even a single rollout
    is valid for SFT-style warm-starting.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrajectoryCollector:
    """
    Collects episode trajectories and exports GRPO-compatible training data.

    Example
    -------
    collector = TrajectoryCollector("trajectories.jsonl")

    obs = env.reset()
    collector.start_episode(tier="easy", episode_id="ep_1")

    while not done:
        action = agent.decide(obs)
        obs, reward, done, info = env.step(action)
        collector.log_step(action=action, observation=obs, reward=reward)

    collector.end_episode(final_score=0.78)
    collector.export_grpo("grpo_data.jsonl")
    """

    def __init__(self, output_path: str = "trajectories.jsonl"):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._current_episode: Optional[Dict] = None
        self._episodes_logged = 0

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def start_episode(self, tier: str, episode_id: str = "") -> None:
        self._current_episode = {
            "episode_id": episode_id or f"ep_{int(time.time() * 1000)}",
            "tier": tier,
            "start_time": time.time(),
            "steps": [],
        }

    def log_step(
        self,
        action: Any,
        observation: Any,
        reward: float,
    ) -> None:
        if self._current_episode is None:
            return
        # Normalise action to dict
        if hasattr(action, "dict"):
            action_dict = action.dict()
        elif hasattr(action, "__dict__"):
            action_dict = action.__dict__
        else:
            action_dict = dict(action) if isinstance(action, dict) else {"raw": str(action)}
        # Normalise observation to dict
        if hasattr(observation, "dict"):
            obs_dict = observation.dict()
        elif hasattr(observation, "__dict__"):
            obs_dict = observation.__dict__
        else:
            obs_dict = dict(observation) if isinstance(observation, dict) else {"raw": str(observation)}

        self._current_episode["steps"].append({
            "step": len(self._current_episode["steps"]) + 1,
            "action": action_dict,
            "observation": obs_dict,
            "reward": round(float(reward), 6),
        })

    def end_episode(self, final_score: float = 0.0) -> Dict:
        if self._current_episode is None:
            return {}
        episode = dict(self._current_episode)
        episode["final_score"] = round(float(final_score), 6)
        episode["total_reward"] = round(sum(s["reward"] for s in episode["steps"]), 6)
        episode["num_steps"] = len(episode["steps"])
        episode["duration_s"] = round(time.time() - episode["start_time"], 2)
        episode.pop("start_time", None)

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode) + "\n")

        self._current_episode = None
        self._episodes_logged += 1
        return episode

    # ── GRPO export ───────────────────────────────────────────────────────────

    def export_grpo(self, output_path: str = "grpo_dataset.jsonl") -> int:
        """
        Convert logged trajectories to GRPO training format.

        Each (observation, action) pair in a trajectory becomes one training
        example. The reward at that step is the scalar signal. For a proper
        GRPO setup you would collect multiple rollouts per state; here we
        format single-rollout data which is suitable for SFT warm-starting
        or single-sample GRPO.

        Returns the number of training examples written.
        """
        out_path = Path(output_path)
        count = 0

        if not self._path.exists():
            print(f"No trajectory file found at {self._path}")
            return 0

        with self._path.open("r", encoding="utf-8") as f_in, \
             out_path.open("w", encoding="utf-8") as f_out:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tier = episode.get("tier", "unknown")
                episode_id = episode.get("episode_id", "")

                for step in episode.get("steps", []):
                    obs = step.get("observation", {})
                    action = step.get("action", {})
                    reward = step.get("reward", 0.0)

                    # Build a text prompt from the observation
                    prompt = _obs_to_prompt(obs)
                    # Build action completion string
                    completion = _action_to_completion(action)

                    grpo_example = {
                        "prompt": prompt,
                        "completions": [completion],  # single rollout
                        "rewards": [reward],
                        "metadata": {
                            "tier": tier,
                            "episode_id": episode_id,
                            "step": step.get("step", 0),
                            "numerical_features": obs.get("metadata", {}).get(
                                "numerical_features", {}
                            ),
                        },
                    }
                    f_out.write(json.dumps(grpo_example) + "\n")
                    count += 1

        print(f"Exported {count} GRPO training examples to {output_path}")
        return count

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return aggregate statistics over all logged episodes."""
        if not self._path.exists():
            return {"episodes": 0}
        episodes = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not episodes:
            return {"episodes": 0}
        scores = [e.get("final_score", 0) for e in episodes]
        by_tier: Dict[str, List[float]] = {}
        for e in episodes:
            tier = e.get("tier", "unknown")
            by_tier.setdefault(tier, []).append(e.get("final_score", 0))
        return {
            "episodes": len(episodes),
            "mean_score": round(sum(scores) / len(scores), 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "by_tier": {t: round(sum(v) / len(v), 4) for t, v in by_tier.items()},
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_to_prompt(obs: dict) -> str:
    meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    features = meta.get("numerical_features", {})
    feature_str = ""
    if features:
        feature_str = (
            f"\n[Features] pressure={features.get('negotiation_pressure', 0):.2f} "
            f"agreement_rate={features.get('clause_agreement_rate', 0):.2f} "
            f"hostility={features.get('vendor_hostility_index', 0):.2f} "
            f"legal_risk={features.get('legal_risk_score', 0):.2f} "
            f"probe_eff={features.get('probe_efficiency', 0):.2f}"
        )
    return (
        f"Clause: {obs.get('clause_id', '')} — {obs.get('clause_text', '')}\n"
        f"Vendor: {obs.get('vendor_response', '')} [{meta.get('vendor_stance', '?')}]\n"
        f"Legal: {obs.get('legal_response', '')} [{meta.get('legal_stance', '?')}]\n"
        f"Probe: {obs.get('probe_result', 'none')}\n"
        f"Round: {obs.get('round_number', 0)}/{obs.get('round_number', 0) + obs.get('rounds_remaining', 0)} "
        f"Agreed: {obs.get('clauses_agreed', 0)}/{obs.get('clauses_total', 0)}"
        f"{feature_str}"
    )


def _action_to_completion(action: dict) -> str:
    action_type = action.get("action_type", "ACCEPT")
    parts = [f'{{"action_type": "{action_type}"']
    if action.get("clause_id"):
        parts.append(f'"clause_id": "{action["clause_id"]}"')
    if action.get("new_text"):
        # Truncate for training efficiency
        text = action["new_text"][:200]
        parts.append(f'"new_text": "{text}"')
    if action.get("party"):
        parts.append(f'"party": "{action["party"]}"')
    if action.get("reason"):
        parts.append(f'"reason": "{action["reason"]}"')
    return ", ".join(parts) + "}"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrajectoryCollector CLI")
    subparsers = parser.add_subparsers(dest="command")

    stats_p = subparsers.add_parser("stats", help="Print stats over a trajectory file")
    stats_p.add_argument("file", help="Path to trajectories JSONL")

    export_p = subparsers.add_parser("export", help="Export GRPO dataset from trajectories")
    export_p.add_argument("input", help="Input trajectories JSONL")
    export_p.add_argument("output", help="Output GRPO JSONL")

    args = parser.parse_args()

    if args.command == "stats":
        c = TrajectoryCollector(args.file)
        s = c.stats()
        print(json.dumps(s, indent=2))

    elif args.command == "export":
        c = TrajectoryCollector(args.input)
        c.export_grpo(args.output)

    else:
        parser.print_help()
