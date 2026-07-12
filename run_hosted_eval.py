"""Run DreamZero-DROID against hosted Cybernetic Physics Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os

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
    parser.add_argument("--request-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--launch-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--keep-session", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.environment_uri:
        raise SystemExit("--environment-uri or CYBERNETICS_DROID_ENV_URI is required")

    try:
        from cybernetics.sim import SimulationClient  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Install a Cybernetics SDK release that provides "
            "cybernetics.sim.SimulationClient and mcp_session"
        ) from exc

    config = HostedDroidConfig(
        environment_uri=args.environment_uri,
        instruction=args.instruction,
        robot_usd_path=args.robot_usd_path,
        max_action_steps=args.max_action_steps,
        open_loop_horizon=args.open_loop_horizon,
        physics_steps_per_action=args.physics_steps_per_action,
        request_timeout_seconds=args.request_timeout_seconds,
        launch_timeout_seconds=args.launch_timeout_seconds,
        readiness_timeout_seconds=args.readiness_timeout_seconds,
        keep_session=args.keep_session,
    )
    sampler = CyberneticsSDKDroidSamplingAPI(
        base_model="dreamzero-droid",
        session_timeout=args.request_timeout_seconds,
    )
    with SimulationClient() as simulation_client:
        result = HostedDroidRunner(simulation_client, sampler, config).run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
