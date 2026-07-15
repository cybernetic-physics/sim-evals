"""Hash-bind an existing hosted DROID evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim_evals.hosted_droid import finalize_hosted_evidence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    manifest = finalize_hosted_evidence_manifest(args.results_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
