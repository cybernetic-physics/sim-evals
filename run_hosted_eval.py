"""Run DreamZero-DROID against hosted Cybernetic Physics Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sim_evals.hosted_droid import HostedDroidConfig, HostedDroidRunner
from sim_evals.inference.cybernetics_dreamzero import CyberneticsSDKDroidSamplingAPI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--physics-steps-per-action", type=int, default=8)
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
    parser.add_argument("--keep-session", action="store_true")
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
    results_dir = args.results_dir or _timestamped_results_dir()

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
        instruction=args.instruction,
        robot_usd_path=args.robot_usd_path,
        max_action_steps=args.max_action_steps,
        open_loop_horizon=args.open_loop_horizon,
        physics_steps_per_action=args.physics_steps_per_action,
        runtime_provider=args.runtime_provider,
        request_timeout_seconds=args.request_timeout_seconds,
        launch_timeout_seconds=args.launch_timeout_seconds,
        readiness_timeout_seconds=args.readiness_timeout_seconds,
        keep_session=args.keep_session,
        results_dir=results_dir,
    )
    sampler = CyberneticsSDKDroidSamplingAPI(
        base_model="dreamzero-droid",
        session_timeout=args.request_timeout_seconds,
    )
    with SimulationClient() as simulation_client:
        result = HostedDroidRunner(simulation_client, sampler, config).run()
    output = result.to_dict()
    output["results_dir"] = str(results_dir)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
