"""Plan or sequentially launch deterministic DROID scene curriculum variants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from sim_evals.curriculum import (
    MAX_TOTAL_VARIANTS,
    MAX_VARIANTS_PER_SPLIT,
    CurriculumError,
    CurriculumPlanConfig,
    CyberneticsWorkflowRunClient,
    launch_curriculum,
    load_or_create_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-environment-uri",
        default=os.environ.get("CYBERNETICS_DROID_ENV_URI"),
        help=(
            "required exact source version URI: "
            "cybernetics://envs/env_.../versions/ver_..."
        ),
    )
    parser.add_argument("--root-seed", type=int, default=20260715)
    parser.add_argument(
        "--train-variants",
        type=int,
        default=8,
        choices=range(1, MAX_VARIANTS_PER_SPLIT + 1),
    )
    parser.add_argument(
        "--validation-variants",
        type=int,
        default=4,
        choices=range(1, MAX_VARIANTS_PER_SPLIT + 1),
    )
    parser.add_argument(
        "--held-out-variants",
        type=int,
        default=4,
        choices=range(1, MAX_VARIANTS_PER_SPLIT + 1),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/droid-curriculum/manifest.json"),
        help="durable plan/execution manifest",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="launch",
        action="store_false",
        help="write the deterministic manifest without creating workflows (default)",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="explicitly create simulation_from_prompt workflows",
    )
    parser.set_defaults(launch=False)
    parser.add_argument(
        "--max-launches",
        type=int,
        default=1,
        choices=range(1, MAX_TOTAL_VARIANTS + 1),
        help="maximum new workflows to create in this invocation (default: 1)",
    )
    parser.add_argument("--workspace-id")
    parser.add_argument("--budget-turns", type=int, default=24, choices=range(2, 65))
    parser.add_argument("--budget-seconds", type=float, default=3600.0)
    parser.add_argument("--workflow-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("CYBERNETICS_API_BASE_URL"),
        help="control-plane API base URL (or CYBERNETICS_API_BASE_URL)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.base_environment_uri:
        raise SystemExit(
            "--base-environment-uri or CYBERNETICS_DROID_ENV_URI is required"
        )
    try:
        config = CurriculumPlanConfig(
            base_environment_uri=args.base_environment_uri,
            root_seed=args.root_seed,
            train_variants=args.train_variants,
            validation_variants=args.validation_variants,
            held_out_variants=args.held_out_variants,
        )
        manifest = load_or_create_manifest(config, args.manifest)
        mode = "dry-run"
        if args.launch:
            mode = "launch"
            with CyberneticsWorkflowRunClient(base_url=args.api_base) as client:
                manifest = launch_curriculum(
                    manifest,
                    client,
                    manifest_path=args.manifest,
                    max_launches=args.max_launches,
                    budget_turns=args.budget_turns,
                    budget_seconds=args.budget_seconds,
                    workflow_timeout_seconds=args.workflow_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                    workspace_id=args.workspace_id,
                )
    except CurriculumError as exc:
        raise SystemExit(str(exc)) from exc

    completed = sum(
        variant.execution.status == "completed" for variant in manifest.variants
    )
    running = sum(
        variant.execution.status == "running" for variant in manifest.variants
    )
    failed = sum(variant.execution.status == "failed" for variant in manifest.variants)
    print(
        json.dumps(
            {
                "mode": mode,
                "manifest": str(args.manifest),
                "base_environment_uri": manifest.base_environment.uri,
                "plan_sha256": manifest.plan_sha256,
                "split_counts": manifest.split_counts,
                "total_variants": len(manifest.variants),
                "completed": completed,
                "running": running,
                "failed": failed,
                "output_environment_uris": list(manifest.output_environment_uris),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
