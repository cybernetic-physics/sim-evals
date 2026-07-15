"""Train a worldline-local DSRL controller against fresh hosted DROID sims.

Every episode launches a new immutable Cybernetic Physics environment version,
creates a new PI0 sampling session, and tears both down before the next episode.
The hosted PI0 base policy remains frozen; only the local DSRL controller and
its replay/checkpoints are mutable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from sim_evals.dsrl import DsrlConfig, TorchDsrlController
from sim_evals.hosted_droid import (
    PI0_DROID_POLICY_PROFILE,
    HostedDroidConfig,
    HostedDroidRunner,
    SimulationClientAPI,
    scene1_cube_in_bowl_success_spec,
)
from sim_evals.inference.cybernetics_dreamzero import (
    CyberneticsSDKDroidSamplingAPI,
)

_MAX_EPISODES_PER_RUN = 1_000
_MAX_ACTION_STEPS_PER_EPISODE = 450
_MAX_UNGATED_TRAIN_EPISODES = 1
_IMMUTABLE_ENVIRONMENT_URI = re.compile(
    r"^cybernetics://envs/[^/?#]+/versions/[^/?#]+$"
)
_EXPECTED_PI0_BASE_POLICY_LINEAGE = dict(PI0_DROID_POLICY_PROFILE)


class _CheckpointableDsrlController(Protocol):
    transitions: int
    updates: int

    @property
    def gamma(self) -> float: ...

    def metadata(self) -> Mapping[str, Any]: ...

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
    ) -> Any: ...

    def record_transition(self, transition: Any) -> Mapping[str, Any]: ...

    def train_after_trajectory(
        self,
        transition_count: int,
    ) -> Mapping[str, Any]: ...

    def save_checkpoint(
        self,
        path: Path,
        *,
        include_replay: bool = True,
    ) -> Mapping[str, Any]: ...


class _CloseOnceSampler:
    """Make sampler ownership explicit across runner and orchestration cleanup."""

    def __init__(self, sampler: Any) -> None:
        self._sampler = sampler
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sampler, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sampler.close()


class _EpisodeBufferedDsrlController:
    """Hold one trajectory fixed, then apply all replay/training work at its end."""

    def __init__(self, controller: _CheckpointableDsrlController) -> None:
        self._controller = controller
        self._transitions: list[Any] = []

    @property
    def gamma(self) -> float:
        return self._controller.gamma

    def metadata(self) -> Mapping[str, Any]:
        return self._controller.metadata()

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
    ) -> Any:
        return self._controller.select_action(
            observation,
            deterministic=deterministic,
        )

    def record_transition(self, transition: Any) -> Mapping[str, Any]:
        self._transitions.append(transition)
        return {
            "mode": "episode_buffered",
            "buffered_transitions": len(self._transitions),
            "transitions": self._controller.transitions,
            "updates": self._controller.updates,
        }

    def flush(self) -> dict[str, Any]:
        transitions_before = self._controller.transitions
        updates_before = self._controller.updates
        buffered_transitions = len(self._transitions)
        for transition in self._transitions:
            self._controller.record_transition(transition)
        self._transitions.clear()
        training = dict(self._controller.train_after_trajectory(buffered_transitions))
        return {
            "buffered_transitions": buffered_transitions,
            "transitions_before": transitions_before,
            "transitions_total": self._controller.transitions,
            "updates_before": updates_before,
            "updates_total": self._controller.updates,
            "updates_delta": self._controller.updates - updates_before,
            "training": training,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-uri",
        default=os.environ.get("CYBERNETICS_DROID_ENV_URI"),
        help=(
            "immutable cybernetics://envs/.../versions/... URI "
            "(or CYBERNETICS_DROID_ENV_URI)"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "training episodes; use 0 only with --resume and one or more "
            "--eval-episodes"
        ),
    )
    parser.add_argument(
        "--allow-zero-success-training",
        action="store_true",
        help=(
            "acknowledge that the strict settled-placement reward may remain zero; "
            "required for more than one training episode"
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="deterministic fresh-session episodes after training; replay is unchanged",
    )
    parser.add_argument(
        "--instruction",
        default=os.environ.get(
            "CYBERNETICS_DROID_INSTRUCTION", "put the cube in the bowl"
        ),
    )
    parser.add_argument(
        "--robot-usd-path",
        default=os.environ.get(
            "CYBERNETICS_DROID_ROBOT_USD",
            "/data/workspace/franka_robotiq_2f_85_flattened.usd",
        ),
    )
    parser.add_argument(
        "--max-action-steps",
        type=int,
        default=200,
        help="primitive-action cap; 200 gives the reference 20 chunks at horizon 10",
    )
    parser.add_argument(
        "--physics-steps-per-action",
        type=int,
        help="override automatic 15 Hz cadence derived from hosted physics dt",
    )
    parser.add_argument("--target-control-hz", type=float, default=15.0)
    parser.add_argument(
        "--runtime-provider",
        choices=("warm_pool", "vast"),
        help="optional Cybernetic Physics simulation runtime",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--launch-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="archive rollout video for every episode (disabled by default)",
    )
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="new run directory (default: UTC timestamp under runs/hosted-dsrl)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="controller checkpoint directory from an earlier run",
    )
    parser.add_argument(
        "--device",
        help="PyTorch device (default: auto; may override device when resuming)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--random-exploration-episodes", type=int)
    parser.add_argument("--initial-updates", type=int)
    parser.add_argument("--updates-per-transition", type=int)
    parser.add_argument(
        "--checkpoint-every-episodes",
        type=int,
        default=10,
        help="write replay-bearing immutable checkpoints at this interval and at end",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="retain at most this many replay-bearing immutable checkpoints",
    )
    return parser


def _timestamped_results_dir(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("runs") / "hosted-dsrl" / name


def _validate_args(args: argparse.Namespace) -> None:
    if not args.environment_uri:
        raise ValueError("--environment-uri or CYBERNETICS_DROID_ENV_URI is required")
    if not _IMMUTABLE_ENVIRONMENT_URI.fullmatch(args.environment_uri):
        raise ValueError(
            "--environment-uri must pin an immutable "
            "cybernetics://envs/.../versions/... version"
        )
    if args.episodes < 0:
        raise ValueError("--episodes must not be negative")
    if args.episodes == 0:
        if args.eval_episodes < 1:
            raise ValueError(
                "--episodes 0 requires at least one --eval-episodes rollout"
            )
        if args.resume is None:
            raise ValueError(
                "--episodes 0 evaluation requires a replay-bearing --resume checkpoint"
            )
    if (
        args.episodes > _MAX_UNGATED_TRAIN_EPISODES
        and not args.allow_zero_success_training
    ):
        raise ValueError(
            "more than one sparse-reward training episode requires "
            "--allow-zero-success-training or an easier curriculum"
        )
    if args.eval_episodes < 0:
        raise ValueError("--eval-episodes must not be negative")
    if args.episodes + args.eval_episodes > _MAX_EPISODES_PER_RUN:
        raise ValueError(f"a run is limited to {_MAX_EPISODES_PER_RUN} total episodes")
    if not 1 <= args.max_action_steps <= _MAX_ACTION_STEPS_PER_EPISODE:
        raise ValueError(
            f"--max-action-steps must be between 1 and {_MAX_ACTION_STEPS_PER_EPISODE}"
        )
    if args.physics_steps_per_action is not None and args.physics_steps_per_action < 1:
        raise ValueError("--physics-steps-per-action must be at least 1")
    for name in (
        "target_control_hz",
        "request_timeout_seconds",
        "launch_timeout_seconds",
        "readiness_timeout_seconds",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be at least 1")
    if args.checkpoint_every_episodes < 1:
        raise ValueError("--checkpoint-every-episodes must be at least 1")
    if args.keep_checkpoints < 1:
        raise ValueError("--keep-checkpoints must be at least 1")
    if args.resume is not None and not args.resume.is_dir():
        raise ValueError("--resume must name an existing checkpoint directory")


def _controller_override_names(args: argparse.Namespace) -> list[str]:
    return [
        name
        for name in (
            "seed",
            "batch_size",
            "replay_capacity",
            "random_exploration_episodes",
            "initial_updates",
            "updates_per_transition",
        )
        if getattr(args, name) is not None
    ]


def _build_controller(args: argparse.Namespace) -> TorchDsrlController:
    if args.resume is not None:
        overrides = _controller_override_names(args)
        if overrides:
            rendered = ", ".join(f"--{name.replace('_', '-')}" for name in overrides)
            raise ValueError(
                f"controller hyperparameters come from --resume; remove {rendered}"
            )
        return TorchDsrlController.load_checkpoint(
            args.resume,
            device=args.device,
            expected_base_policy_metadata=_EXPECTED_PI0_BASE_POLICY_LINEAGE,
            require_replay=True,
        )

    defaults = DsrlConfig()
    config = DsrlConfig(
        seed=defaults.seed if args.seed is None else args.seed,
        device=defaults.device if args.device is None else args.device,
        batch_size=(
            defaults.batch_size if args.batch_size is None else args.batch_size
        ),
        replay_capacity=(
            defaults.replay_capacity
            if args.replay_capacity is None
            else args.replay_capacity
        ),
        random_exploration_episodes=(
            defaults.random_exploration_episodes
            if args.random_exploration_episodes is None
            else args.random_exploration_episodes
        ),
        initial_updates=(
            defaults.initial_updates
            if args.initial_updates is None
            else args.initial_updates
        ),
        updates_per_transition=(
            defaults.updates_per_transition
            if args.updates_per_transition is None
            else args.updates_per_transition
        ),
    )
    return TorchDsrlController(config)


def run_training(
    args: argparse.Namespace,
    *,
    simulation_client: SimulationClientAPI,
    controller: _CheckpointableDsrlController,
    sampler_factory: Callable[[], Any],
    runner_factory: Callable[..., Any] = HostedDroidRunner,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the bounded plan, keeping controller state but no hosted session state."""

    _validate_args(args)
    _validate_controller_lineage(controller)
    clock = now or (lambda: datetime.now(timezone.utc))
    results_dir = args.results_dir or _timestamped_results_dir(clock())
    episodes_path = results_dir / "episodes.jsonl"
    manifest_path = results_dir / "train-manifest.json"
    if results_dir.exists():
        if not results_dir.is_dir() or any(results_dir.iterdir()):
            raise ValueError(f"results directory must be empty: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    created_at = _utc_now(clock)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "environment_uri": args.environment_uri,
        "base_model": "pi0-droid",
        "base_policy_frozen": True,
        "expected_base_policy_lineage": dict(_EXPECTED_PI0_BASE_POLICY_LINEAGE),
        "task_success_predicate": "scene1-cube-in-bowl",
        "open_loop_horizon": 10,
        "reward_contract": {
            "kind": "sparse_chunk",
            "failure_reward": -1.0,
            "success_reward": 0.0,
            "strict_settled_placement": True,
            "allow_zero_success_training": args.allow_zero_success_training,
        },
        "planned_train_episodes": args.episodes,
        "planned_eval_episodes": args.eval_episodes,
        "completed_train_episodes": 0,
        "completed_eval_episodes": 0,
        "max_action_steps": args.max_action_steps,
        "latest_checkpoint": str(args.resume) if args.resume is not None else None,
        "latest_lightweight_checkpoint": None,
        "latest_full_checkpoint": (
            str(args.resume) if args.resume is not None else None
        ),
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "checkpoint_policy": {
            "lightweight_latest_every_train_episode": True,
            "lightweight_latest_role": "inspection_or_evaluation_only",
            "training_resume_requires_full_replay": True,
            "full_replay_every_episodes": args.checkpoint_every_episodes,
            "full_replay_at_final_episode": True,
            "retained_full_checkpoints": args.keep_checkpoints,
        },
        "retained_full_checkpoints": [],
        "controller": dict(controller.metadata()),
    }
    _atomic_json(manifest_path, manifest)

    plan = [("train", index) for index in range(args.episodes)]
    plan.extend(("eval", index) for index in range(args.eval_episodes))
    global_index = 0
    for phase, phase_index in plan:
        episode_dir = results_dir / "episodes" / f"{global_index:06d}"
        started_at = _utc_now(clock)
        record: dict[str, Any] = {
            "schema_version": 1,
            "episode_index": global_index,
            "phase": phase,
            "phase_episode_index": phase_index,
            "status": "running",
            "started_at": started_at,
            "environment_uri": args.environment_uri,
            "results_dir": str(episode_dir),
        }
        try:
            sampler = _CloseOnceSampler(sampler_factory())
            episode_controller = (
                _EpisodeBufferedDsrlController(controller)
                if phase == "train"
                else controller
            )
            try:
                config = HostedDroidConfig(
                    environment_uri=args.environment_uri,
                    session_id=None,
                    base_model="pi0-droid",
                    instruction=args.instruction,
                    robot_usd_path=args.robot_usd_path,
                    max_action_steps=args.max_action_steps,
                    open_loop_horizon=10,
                    physics_steps_per_action=args.physics_steps_per_action,
                    target_control_hz=args.target_control_hz,
                    runtime_provider=args.runtime_provider,
                    policy_mode="native",
                    include_predicted_video=False,
                    request_timeout_seconds=args.request_timeout_seconds,
                    launch_timeout_seconds=args.launch_timeout_seconds,
                    readiness_timeout_seconds=args.readiness_timeout_seconds,
                    keep_session=False,
                    record_video=args.record_video,
                    video_fps=args.video_fps,
                    results_dir=episode_dir,
                    task_success=scene1_cube_in_bowl_success_spec(),
                )
                runner = runner_factory(
                    simulation_client,
                    sampler,
                    config,
                    dsrl_controller=episode_controller,
                    deterministic_dsrl=phase == "eval",
                    train_dsrl_controller=phase == "train",
                )
                result = runner.run()
            finally:
                sampler.close()
            result_payload = result.to_dict()
            if phase == "train":
                assert isinstance(
                    episode_controller,
                    _EpisodeBufferedDsrlController,
                )
                training_metrics = episode_controller.flush()
                result_payload["dsrl_updates"] = training_metrics["updates_total"]
                result_payload["dsrl_updates_total"] = training_metrics["updates_total"]
                result_payload["dsrl_updates_delta"] = training_metrics["updates_delta"]
                record["post_episode_training"] = training_metrics
                controller_dir = results_dir / "controller"
                latest_dir = controller_dir / "latest"
                latest_checkpoint = controller.save_checkpoint(
                    latest_dir,
                    include_replay=False,
                )
                checkpoints: dict[str, Any] = {
                    "latest": {
                        "path": str(latest_dir),
                        "includes_replay": False,
                        "manifest": dict(latest_checkpoint),
                    }
                }
                manifest["latest_lightweight_checkpoint"] = str(latest_dir)
                train_episode_number = phase_index + 1
                full_checkpoint_due = (
                    train_episode_number % args.checkpoint_every_episodes == 0
                    or train_episode_number == args.episodes
                )
                if full_checkpoint_due:
                    full_dir = controller_dir / f"checkpoint-{train_episode_number:06d}"
                    full_checkpoint = controller.save_checkpoint(
                        full_dir,
                        include_replay=True,
                    )
                    pruned = _prune_full_checkpoints(
                        controller_dir,
                        keep=args.keep_checkpoints,
                    )
                    checkpoints["full"] = {
                        "path": str(full_dir),
                        "includes_replay": True,
                        "manifest": dict(full_checkpoint),
                        "pruned": pruned,
                    }
                    manifest["latest_checkpoint"] = str(full_dir)
                    manifest["latest_full_checkpoint"] = str(full_dir)
                    manifest["retained_full_checkpoints"] = [
                        str(path) for path in _full_checkpoint_paths(controller_dir)
                    ]
                record["checkpoints"] = checkpoints
            record["result"] = result_payload
            record["status"] = "succeeded"
            record["finished_at"] = _utc_now(clock)
            _append_jsonl(episodes_path, record)
            if phase == "train":
                manifest["completed_train_episodes"] += 1
            else:
                manifest["completed_eval_episodes"] += 1
            manifest["controller"] = dict(controller.metadata())
            manifest["updated_at"] = record["finished_at"]
            _atomic_json(manifest_path, manifest)
        except BaseException as exc:
            record["status"] = "failed"
            record["finished_at"] = _utc_now(clock)
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _append_jsonl(episodes_path, record)
            manifest["status"] = "failed"
            manifest["updated_at"] = record["finished_at"]
            manifest["error"] = dict(record["error"])
            manifest["controller"] = dict(controller.metadata())
            _atomic_json(manifest_path, manifest)
            raise
        global_index += 1

    manifest["status"] = "succeeded"
    manifest["updated_at"] = _utc_now(clock)
    manifest["controller"] = dict(controller.metadata())
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_controller_lineage(
    controller: _CheckpointableDsrlController,
) -> None:
    metadata = controller.metadata()
    raw_lineage = metadata.get("base_policy_metadata")
    if raw_lineage is None:
        return
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("controller base-policy metadata must be a mapping")
    if dict(raw_lineage) != _EXPECTED_PI0_BASE_POLICY_LINEAGE:
        raise ValueError("controller does not match the pinned PI0-DROID base policy")


def _full_checkpoint_paths(controller_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in controller_dir.glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )


def _prune_full_checkpoints(controller_dir: Path, *, keep: int) -> list[str]:
    checkpoints = _full_checkpoint_paths(controller_dir)
    pruned: list[str] = []
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)
        pruned.append(str(path))
    return pruned


def _utc_now(clock: Callable[[], datetime]) -> str:
    return clock().astimezone(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temp_path,
        os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
        0o600,
    )
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError(f"write made no progress for {path}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temp_path, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError(f"write made no progress for {path}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    args = _parser().parse_args()
    try:
        _validate_args(args)
        controller = _build_controller(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        from cybernetics.sim import SimulationClient  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Install a Cybernetics SDK release that provides "
            "cybernetics.sim.SimulationClient and mcp_session"
        ) from exc

    def sampler_factory() -> CyberneticsSDKDroidSamplingAPI:
        return CyberneticsSDKDroidSamplingAPI(
            base_model="pi0-droid",
            session_timeout=args.request_timeout_seconds,
            policy_mode="native",
            include_predicted_video=False,
        )

    with SimulationClient() as simulation_client:
        manifest = run_training(
            args,
            simulation_client=cast(SimulationClientAPI, simulation_client),
            controller=controller,
            sampler_factory=sampler_factory,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
