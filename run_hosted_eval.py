"""Run a Cybernetics DROID policy against hosted Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from sim_evals.hosted_droid import (
    HostedDroidConfig,
    HostedDroidRunner,
    SimulationClientAPI,
    scene1_cube_in_bowl_success_spec,
)
from sim_evals.inference.cybernetics_dreamzero import CyberneticsSDKDroidSamplingAPI


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
    parser.add_argument("--max-action-steps", type=int, default=450)
    parser.add_argument("--open-loop-horizon", type=int)
    parser.add_argument(
        "--physics-steps-per-action",
        type=int,
        help="override automatic 15 Hz cadence derived from the hosted physics dt",
    )
    parser.add_argument("--target-control-hz", type=float, default=15.0)
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


def main() -> None:
    args = _parser().parse_args()
    if not args.environment_uri:
        raise SystemExit("--environment-uri or CYBERNETICS_DROID_ENV_URI is required")
    if args.base_model == "pi0-droid" and args.policy_mode != "native":
        raise SystemExit("pi0-droid supports only --policy-mode native")
    if args.base_model == "pi0-droid" and args.include_predicted_video:
        raise SystemExit("pi0-droid does not produce predicted video")
    open_loop_horizon = args.open_loop_horizon
    if open_loop_horizon is None:
        open_loop_horizon = 10 if args.base_model == "pi0-droid" else 8
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

    config = HostedDroidConfig(
        environment_uri=args.environment_uri,
        session_id=args.session_id,
        base_model=args.base_model,
        instruction=args.instruction,
        robot_usd_path=args.robot_usd_path,
        max_action_steps=args.max_action_steps,
        open_loop_horizon=open_loop_horizon,
        physics_steps_per_action=args.physics_steps_per_action,
        target_control_hz=args.target_control_hz,
        runtime_provider=args.runtime_provider,
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
    sampler = CyberneticsSDKDroidSamplingAPI(
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
