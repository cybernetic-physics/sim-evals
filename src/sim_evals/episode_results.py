"""Durable JSONL and JSON episode result output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class EpisodeResultWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "episodes.jsonl"
        self.json_path = self.run_dir / "episodes.json"
        self._results: list[dict[str, Any]] = []
        self.jsonl_path.write_text("")
        self._write_json()

    def record(self, result: Mapping[str, Any]) -> None:
        serialized = dict(result)
        self._results.append(serialized)
        with self.jsonl_path.open("a") as output:
            output.write(json.dumps(serialized, sort_keys=True) + "\n")
        self._write_json()

    def _write_json(self) -> None:
        self.json_path.write_text(
            json.dumps(self._results, indent=2, sort_keys=True) + "\n"
        )
