"""Recover a hosted DROID MP4 from already-persisted post-action frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim_evals.hosted_droid import recover_hosted_video_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    recovery = recover_hosted_video_evidence(args.results_dir)
    print(json.dumps(recovery, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
