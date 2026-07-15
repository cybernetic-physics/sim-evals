"""Run a Cybernetics DROID policy against hosted Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

from sim_evals.hosted_droid import (
    HostedDroidConfig,
    HostedDroidRunner,
    SimulationClientAPI,
    _droid_joint_positions_for_policy_action,
    _validate_policy_response,
    scene1_cube_in_bowl_success_spec,
)
from sim_evals.inference.cybernetics_dreamzero import (
    CyberneticsSDKDroidSamplingAPI,
    _action_chunk,
)


class _RecordedPi0Replay:
    """Replay a verified applied PI0 prefix without calling Worldlines."""

    action_source = "recorded_replay"

    def __init__(
        self,
        *,
        samples: tuple[dict[str, object], ...],
        source_sha256: str,
        open_loop_horizon: int,
        applied_action_steps: int,
    ) -> None:
        self._samples = samples
        self.source_sha256 = source_sha256
        self.open_loop_horizon = open_loop_horizon
        self.applied_action_steps = applied_action_steps
        self._index = 0
        self._closed = False

    @classmethod
    def load(cls, evidence_dir: Path) -> "_RecordedPi0Replay":
        source_dir = evidence_dir.expanduser().resolve()
        config_path = source_dir / "config.json"
        actions_path = source_dir / "actions.jsonl"
        config_payload = _json_object(config_path)
        source_config = config_payload.get("config")
        if (
            type(config_payload.get("schema_version")) is not int
            or config_payload.get("schema_version") != 9
            or not isinstance(source_config, dict)
        ):
            raise ValueError("recorded replay requires schema-v9 config evidence")
        if source_config.get("base_model") != "pi0-droid":
            raise ValueError("recorded replay requires pi0-droid evidence")
        open_loop_horizon = source_config.get("open_loop_horizon")
        if (
            isinstance(open_loop_horizon, bool)
            or not isinstance(open_loop_horizon, int)
            or open_loop_horizon < 1
        ):
            raise ValueError("recorded replay has an invalid open-loop horizon")

        raw = actions_path.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        records = _jsonl_objects(raw, actions_path)
        samples: list[dict[str, object]] = []
        sampled_chunks: list[np.ndarray] = []
        targets: list[dict[str, object]] = []
        applied: list[dict[str, object]] = []
        for record in records:
            record_type = record.get("record_type")
            if record_type == "sample":
                if len(targets) != len(applied):
                    raise ValueError(
                        "recorded sample cannot interrupt a pending action target"
                    )
                sample_index = record.get("sample_index")
                if type(sample_index) is not int or sample_index != len(samples):
                    raise ValueError(
                        "recorded action samples must have contiguous zero-based indices"
                    )
                if len(applied) != sample_index * open_loop_horizon:
                    raise ValueError(
                        "recorded samples must begin at an action-chunk boundary"
                    )
                response = _replay_sample_response(record, source_sha256)
                sampled_chunk = _action_chunk(response)
                _validate_policy_response("pi0-droid", response, sampled_chunk)
                executed_chunk = _float_array(
                    record.get("action_chunk"),
                    "recorded sample execution slice",
                )
                if not np.array_equal(
                    executed_chunk,
                    sampled_chunk[:open_loop_horizon],
                ):
                    raise ValueError(
                        "recorded sample execution slice does not match its PI0 chunk"
                    )
                samples.append(response)
                sampled_chunks.append(sampled_chunk)
                continue
            if record_type == "action_target":
                if len(targets) != len(applied):
                    raise ValueError(
                        "recorded action target must follow its prior applied action"
                    )
                _validate_replay_action_record(
                    record,
                    expected_action_index=len(targets),
                    sampled_chunks=sampled_chunks,
                    open_loop_horizon=open_loop_horizon,
                )
                targets.append(record)
                continue
            if record_type == "applied_action":
                if len(targets) != len(applied) + 1:
                    raise ValueError(
                        "recorded applied action must follow exactly one target"
                    )
                _validate_replay_action_record(
                    record,
                    expected_action_index=len(applied),
                    sampled_chunks=sampled_chunks,
                    open_loop_horizon=open_loop_horizon,
                )
                if len(applied) >= len(targets) or not _same_replay_action(
                    record, targets[len(applied)]
                ):
                    raise ValueError(
                        "recorded applied action does not match its accepted target"
                    )
                applied.append(record)
                continue
            raise ValueError(f"recorded replay has unknown record type {record_type!r}")

        if not samples:
            raise ValueError("recorded replay contains no sample records")
        if not applied or len(targets) != len(applied):
            raise ValueError(
                "recorded replay requires a complete applied-action prefix"
            )
        expected_samples = (len(applied) + open_loop_horizon - 1) // open_loop_horizon
        if len(samples) != expected_samples:
            raise ValueError(
                "recorded replay samples must exactly cover its applied-action prefix"
            )
        return cls(
            samples=tuple(samples),
            source_sha256=source_sha256,
            open_loop_horizon=open_loop_horizon,
            applied_action_steps=len(applied),
        )

    def reset_sampling_session(self) -> None:
        if self._closed:
            raise RuntimeError("recorded action replay is closed")
        self._index = 0

    def sample_droid(self, observation: object, *, timeout: float) -> dict[str, object]:
        del observation, timeout
        if self._closed:
            raise RuntimeError("recorded action replay is closed")
        if self._index >= len(self._samples):
            raise RuntimeError("recorded action replay exhausted its sample chunks")
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def close(self) -> None:
        self._closed = True


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recorded replay could not read {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"recorded replay {path.name} must contain an object")
    return payload


def _jsonl_objects(raw: bytes, path: Path) -> list[dict[str, object]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError(f"recorded replay could not decode {path.name}") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"recorded replay {path.name} line {line_number} is invalid"
            ) from exc
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 9
        ):
            raise ValueError(
                f"recorded replay {path.name} line {line_number} is not schema v9"
            )
        records.append(record)
    return records


def _replay_sample_response(
    record: dict[str, object], source_sha256: str
) -> dict[str, object]:
    metadata = record.get("policy_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("recorded PI0 sample is missing policy metadata")
    return {
        "action_chunk": record.get("sampled_action_chunk"),
        "policy_metadata": {
            **metadata,
            "evaluation_action_source": "recorded_replay",
            "replay_source_sha256": source_sha256,
        },
    }


def _validate_replay_action_record(
    record: dict[str, object],
    *,
    expected_action_index: int,
    sampled_chunks: list[np.ndarray],
    open_loop_horizon: int,
) -> None:
    action_index = record.get("action_index")
    if type(action_index) is not int or action_index != expected_action_index:
        raise ValueError("recorded actions must have contiguous zero-based indices")
    sample_index = record.get("sample_index")
    chunk_index = record.get("chunk_index")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
        or sample_index >= len(sampled_chunks)
        or isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
        or chunk_index >= open_loop_horizon
    ):
        raise ValueError("recorded action has an invalid sample/chunk index")
    expected_sample_index, expected_chunk_index = divmod(
        expected_action_index, open_loop_horizon
    )
    if (sample_index, chunk_index) != (expected_sample_index, expected_chunk_index):
        raise ValueError(
            "recorded action sample/chunk slot does not match its action index"
        )
    policy_action = _float_array(record.get("policy_action"), "recorded policy action")
    if not np.array_equal(policy_action, sampled_chunks[sample_index][chunk_index]):
        raise ValueError("recorded action does not match its sampled PI0 slot")
    expected_joint_positions = np.asarray(
        _droid_joint_positions_for_policy_action(policy_action),
        dtype=np.float64,
    )
    recorded_joint_positions = _float_array(
        record.get("joint_positions"),
        "recorded joint positions",
        dtype=np.float64,
    )
    if (
        recorded_joint_positions.shape != (8,)
        or not np.array_equal(
            recorded_joint_positions[:7].astype(np.float32),
            policy_action[:7],
        )
        or recorded_joint_positions[7] != expected_joint_positions[7]
    ):
        raise ValueError("recorded joint target does not match its DROID policy action")
    joint_indices = record.get("joint_indices")
    if (
        not isinstance(joint_indices, list)
        or len(joint_indices) != 8
        or any(type(value) is not int or value < 0 for value in joint_indices)
        or len(set(joint_indices)) != 8
    ):
        raise ValueError("recorded joint indices must be eight distinct integers")
    if record.get("record_type") == "applied_action":
        _validate_simulation_timing(record.get("simulation_timing"))


def _float_array(
    value: object,
    name: str,
    *,
    dtype: object = np.float32,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"recorded simulation timing {name} must be {qualifier}")
    return float(value)


def _validate_simulation_timing(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("recorded applied action requires simulation timing")
    stepped = value.get("stepped")
    if type(stepped) is not int or stepped < 1:
        raise ValueError("recorded simulation timing stepped must be positive integer")
    before = value.get("before")
    after = value.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("recorded simulation timing requires before and after states")
    before_time = _finite_number(before.get("current_time"), "before.current_time")
    after_time = _finite_number(after.get("current_time"), "after.current_time")
    before_dt = _finite_number(
        before.get("physics_dt"), "before.physics_dt", positive=True
    )
    after_dt = _finite_number(
        after.get("physics_dt"), "after.physics_dt", positive=True
    )
    expected = _finite_number(
        value.get("expected_simulation_seconds"),
        "expected_simulation_seconds",
        positive=True,
    )
    observed = _finite_number(
        value.get("observed_simulation_seconds"),
        "observed_simulation_seconds",
        positive=True,
    )
    drift = _finite_number(
        value.get("timeline_drift_seconds"), "timeline_drift_seconds"
    )
    if (
        before.get("timeline_state") != "paused"
        or after.get("timeline_state") != "paused"
    ):
        raise ValueError("recorded simulation timing must begin and end paused")
    if value.get("joint_target_control_source") != "runtime_articulation":
        raise ValueError(
            "recorded simulation timing requires runtime articulation control"
        )
    tolerance = max(1e-9, before_dt * 1e-6)
    if not np.isclose(before_dt, after_dt, rtol=0.0, atol=tolerance):
        raise ValueError("recorded simulation timing physics dt changed")
    if not np.isclose(expected, stepped * before_dt, rtol=0.0, atol=tolerance):
        raise ValueError("recorded simulation timing expected duration is inconsistent")
    if not np.isclose(observed, after_time - before_time, rtol=0.0, atol=tolerance):
        raise ValueError("recorded simulation timing observed duration is inconsistent")
    if not np.isclose(drift, observed - expected, rtol=0.0, atol=tolerance):
        raise ValueError("recorded simulation timing drift is inconsistent")


def _same_replay_action(applied: dict[str, object], target: dict[str, object]) -> bool:
    return all(
        applied.get(field) == target.get(field)
        for field in (
            "action_index",
            "sample_index",
            "chunk_index",
            "policy_action",
            "joint_positions",
            "joint_indices",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=os.environ.get("CYBERNETICS_DROID_BASE_MODEL", "dreamzero-droid"),
        choices=("dreamzero-droid", "pi0-droid"),
    )
    parser.add_argument(
        "--environment-uri",
        default=os.environ.get("CYBERNETICS_DROID_ENV_URI"),
        help="cybernetics:// environment URI (or CYBERNETICS_DROID_ENV_URI)",
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
    parser.add_argument("--max-action-steps", type=int)
    parser.add_argument("--open-loop-horizon", type=int)
    parser.add_argument(
        "--physics-steps-per-action",
        type=int,
        help="override automatic 15 Hz cadence derived from the hosted physics dt",
    )
    parser.add_argument("--target-control-hz", type=float, default=15.0)
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=240.0,
        help="fixed PhysX update rate; must be an integer multiple of control Hz",
    )
    parser.add_argument("--solver-position-iterations", type=int, default=64)
    parser.add_argument("--solver-velocity-iterations", type=int, default=1)
    parser.add_argument(
        "--task-success-predicate",
        choices=("scene1-cube-in-bowl",),
        help=(
            "opt in to policy-lift, geometric placement, release, and settled-state "
            "acceptance for the immutable DROID scene"
        ),
    )
    parser.add_argument("--policy-mode", choices=("native", "sde"), default="native")
    parser.add_argument(
        "--include-predicted-video",
        action="store_true",
        help="request and archive DreamZero's bounded future-video prediction",
    )
    parser.add_argument(
        "--runtime-provider",
        choices=("warm_pool", "vast"),
        help=(
            "optional Cybernetic Physics simulation runtime; vast requires a "
            "service or system-admin credential"
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "resume an existing Cybernetic Physics session instead of launching "
            "another runtime; attached sessions remain caller-owned"
        ),
    )
    parser.add_argument(
        "--replay-evidence-dir",
        type=Path,
        help=(
            "physics-control mode: replay the verified applied prefix from a "
            "prior schema-v9 evidence directory without calling Worldlines"
        ),
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--launch-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=600.0)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument(
        "--keep-session",
        dest="keep_session",
        action="store_true",
        help="retain the hosted session after evaluation (default)",
    )
    lifecycle.add_argument(
        "--stop-session",
        dest="keep_session",
        action="store_false",
        help="explicitly stop a session launched by this evaluator after evaluation",
    )
    parser.set_defaults(keep_session=True)
    video = parser.add_mutually_exclusive_group()
    video.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        help="save an action-by-action rollout MP4 (default)",
    )
    video.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="disable rollout MP4 capture",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "local evidence directory (default: a UTC timestamp under "
            "runs/hosted-droid)"
        ),
    )
    return parser


def _timestamped_results_dir(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("runs") / "hosted-droid" / name


def _resolve_open_loop_horizon(requested: int | None) -> int:
    return 8 if requested is None else requested


def main() -> None:
    args = _parser().parse_args()
    if not args.environment_uri:
        raise SystemExit("--environment-uri or CYBERNETICS_DROID_ENV_URI is required")
    if args.base_model == "pi0-droid" and args.policy_mode != "native":
        raise SystemExit("pi0-droid supports only --policy-mode native")
    if args.base_model == "pi0-droid" and args.include_predicted_video:
        raise SystemExit("pi0-droid does not produce predicted video")
    if args.replay_evidence_dir is not None and args.session_id is not None:
        raise SystemExit("recorded replay requires a freshly launched session")
    if args.replay_evidence_dir is not None and args.keep_session:
        raise SystemExit("recorded replay requires --stop-session")
    open_loop_horizon = _resolve_open_loop_horizon(args.open_loop_horizon)
    results_dir = args.results_dir or _timestamped_results_dir()
    task_success = (
        scene1_cube_in_bowl_success_spec()
        if args.task_success_predicate == "scene1-cube-in-bowl"
        else None
    )

    try:
        from cybernetics.sim import SimulationClient  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Install a Cybernetics SDK release that provides "
            "cybernetics.sim.SimulationClient and mcp_session"
        ) from exc

    replay_sampler = (
        _RecordedPi0Replay.load(args.replay_evidence_dir)
        if args.replay_evidence_dir is not None
        else None
    )
    if replay_sampler is not None:
        if args.base_model != "pi0-droid":
            raise SystemExit("recorded replay requires --base-model pi0-droid")
        if open_loop_horizon != replay_sampler.open_loop_horizon:
            raise SystemExit(
                "recorded replay open-loop horizon does not match its source"
            )
        if args.max_action_steps not in {
            None,
            replay_sampler.applied_action_steps,
        }:
            raise SystemExit(
                "recorded replay action limit must equal its verified applied prefix"
            )
        max_action_steps = replay_sampler.applied_action_steps
    else:
        max_action_steps = (
            args.max_action_steps if args.max_action_steps is not None else 450
        )
    config = HostedDroidConfig(
        environment_uri=args.environment_uri,
        session_id=args.session_id,
        base_model=args.base_model,
        instruction=args.instruction,
        robot_usd_path=args.robot_usd_path,
        max_action_steps=max_action_steps,
        open_loop_horizon=open_loop_horizon,
        physics_steps_per_action=args.physics_steps_per_action,
        target_control_hz=args.target_control_hz,
        physics_hz=args.physics_hz,
        solver_position_iterations=args.solver_position_iterations,
        solver_velocity_iterations=args.solver_velocity_iterations,
        runtime_provider=args.runtime_provider,
        action_source=(
            "recorded_replay" if replay_sampler is not None else "worldlines_policy"
        ),
        replay_source_sha256=(
            replay_sampler.source_sha256 if replay_sampler is not None else None
        ),
        policy_mode=args.policy_mode,
        include_predicted_video=args.include_predicted_video,
        request_timeout_seconds=args.request_timeout_seconds,
        launch_timeout_seconds=args.launch_timeout_seconds,
        readiness_timeout_seconds=args.readiness_timeout_seconds,
        keep_session=args.keep_session,
        record_video=args.record_video,
        video_fps=args.video_fps,
        results_dir=results_dir,
        task_success=task_success,
    )
    sampler = replay_sampler or CyberneticsSDKDroidSamplingAPI(
        base_model=args.base_model,
        session_timeout=args.request_timeout_seconds,
        policy_mode=args.policy_mode,
        include_predicted_video=args.include_predicted_video,
    )
    with SimulationClient() as simulation_client:
        result = HostedDroidRunner(
            cast(SimulationClientAPI, simulation_client),
            sampler,
            config,
        ).run()
    output = result.to_dict()
    output["results_dir"] = str(results_dir)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
