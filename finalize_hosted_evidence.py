"""Hash-bind an existing hosted DROID evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim_evals.hosted_droid import finalize_hosted_evidence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--artifact-revision",
        help="sim-evals Git revision that produced the existing artifacts",
    )
    parser.add_argument("--artifact-sim-evals-version")
    parser.add_argument("--artifact-sdk-version")
    args = parser.parse_args()
    if not args.artifact_revision and (
        args.artifact_sim_evals_version or args.artifact_sdk_version
    ):
        parser.error("artifact versions require --artifact-revision")
    artifact_producer = None
    if args.artifact_revision:
        artifact_producer = {
            "status": "known",
            "sim_evals_revision": args.artifact_revision,
            "sim_evals_version": args.artifact_sim_evals_version,
            "cybernetics_sdk_version": args.artifact_sdk_version,
        }
    manifest = finalize_hosted_evidence_manifest(
        args.results_dir,
        artifact_producer=artifact_producer,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
