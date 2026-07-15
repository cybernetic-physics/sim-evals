from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from run_hosted_eval import _RecordedPi0Replay, _parser


def _pi0_sample(sample_index: int) -> dict[str, object]:
    sampled = [
        [
            float(sample_index * 100 + chunk_index * 10 + value_index) / 1000.0
            for value_index in range(8)
        ]
        for chunk_index in range(10)
    ]
    return {
        "schema_version": 9,
        "record_type": "sample",
        "sample_index": sample_index,
        "sampled_action_chunk": sampled,
        "action_chunk": sampled[:8],
        "policy_metadata": {
            "base_model": "pi0-droid",
            "openpi_config": "pi0_droid_jointpos_polaris",
            "checkpoint_uri": (
                "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris"
            ),
            "openpi_source_commit": "714ec9aa5e4e9b73b98c6bf3a328f377268e26f9",
            "action_space": "droid_joint_position",
            "action_horizon": 10,
            "action_dim": 8,
        },
    }


def _write_replay_evidence(
    directory: Path,
    *,
    applied_action_steps: int = 10,
    first_sample_index: int = 0,
) -> None:
    config = {
        "schema_version": 9,
        "config": {
            "base_model": "pi0-droid",
            "open_loop_horizon": 8,
        },
    }
    (directory / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    records: list[dict[str, object]] = []
    action_index = 0
    sample_index = first_sample_index
    while action_index < applied_action_steps:
        sample = _pi0_sample(sample_index)
        records.append(sample)
        chunk = sample["sampled_action_chunk"]
        for chunk_index in range(8):
            if action_index >= applied_action_steps:
                break
            policy_action = chunk[chunk_index]
            gripper = math.pi / 4 if float(policy_action[7]) > 0.5 else 0.0
            physics_dt = 1.0 / 120.0
            stepped = 8
            before_time = action_index * stepped * physics_dt
            expected_seconds = stepped * physics_dt
            common = {
                "schema_version": 9,
                "action_index": action_index,
                "sample_index": sample_index,
                "chunk_index": chunk_index,
                "policy_action": policy_action,
                "joint_positions": [*policy_action[:7], gripper],
                "joint_indices": list(range(8)),
            }
            records.append({**common, "record_type": "action_target"})
            records.append(
                {
                    **common,
                    "record_type": "applied_action",
                    "simulation_timing": {
                        "before": {
                            "current_time": before_time,
                            "physics_dt": physics_dt,
                            "timeline_state": "paused",
                        },
                        "after": {
                            "current_time": before_time + expected_seconds,
                            "physics_dt": physics_dt,
                            "timeline_state": "paused",
                        },
                        "stepped": stepped,
                        "expected_simulation_seconds": expected_seconds,
                        "observed_simulation_seconds": expected_seconds,
                        "timeline_drift_seconds": 0.0,
                        "joint_target_control_source": "runtime_articulation",
                    },
                }
            )
            action_index += 1
        sample_index += 1
    (directory / "actions.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class HostedEvalParserTests(unittest.TestCase):
    def test_keeps_launched_session_by_default(self) -> None:
        args = _parser().parse_args([])
        self.assertTrue(args.keep_session)
        self.assertEqual(args.physics_hz, 240.0)

    def test_stop_session_is_explicit(self) -> None:
        args = _parser().parse_args(["--stop-session"])
        self.assertFalse(args.keep_session)

    def test_keep_session_remains_compatible(self) -> None:
        args = _parser().parse_args(["--keep-session"])
        self.assertTrue(args.keep_session)

    def test_scene1_task_success_predicate_is_explicit(self) -> None:
        default_args = _parser().parse_args([])
        selected_args = _parser().parse_args(
            ["--task-success-predicate", "scene1-cube-in-bowl"]
        )

        self.assertIsNone(default_args.task_success_predicate)
        self.assertEqual(
            selected_args.task_success_predicate,
            "scene1-cube-in-bowl",
        )

    def test_physics_replay_controls_are_explicit(self) -> None:
        args = _parser().parse_args(
            [
                "--physics-hz",
                "240",
                "--solver-position-iterations",
                "64",
                "--solver-velocity-iterations",
                "1",
                "--replay-evidence-dir",
                "evidence",
            ]
        )

        self.assertEqual(args.physics_hz, 240.0)
        self.assertEqual(args.solver_position_iterations, 64)
        self.assertEqual(args.solver_velocity_iterations, 1)
        self.assertEqual(args.replay_evidence_dir, Path("evidence"))

    def test_recorded_sampler_replays_full_pi0_chunks_and_records_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source)
            sampler = _RecordedPi0Replay.load(source)

            sampler.reset_sampling_session()
            first = sampler.sample_droid(object(), timeout=1.0)
            second = sampler.sample_droid(object(), timeout=1.0)

            self.assertEqual(len(first["action_chunk"]), 10)
            self.assertEqual(len(second["action_chunk"]), 10)
            self.assertEqual(len(sampler.source_sha256), 64)
            self.assertEqual(sampler.open_loop_horizon, 8)
            self.assertEqual(sampler.applied_action_steps, 10)
            metadata = first["policy_metadata"]
            self.assertEqual(metadata["evaluation_action_source"], "recorded_replay")
            self.assertEqual(metadata["replay_source_sha256"], sampler.source_sha256)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                sampler.sample_droid(object(), timeout=1.0)

    def test_recorded_sampler_rejects_non_contiguous_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, first_sample_index=1)

            with self.assertRaisesRegex(ValueError, "contiguous"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_derives_exact_45_action_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=45)

            sampler = _RecordedPi0Replay.load(source)
            samples = [sampler.sample_droid(object(), timeout=1.0) for _ in range(6)]

            self.assertEqual(sampler.applied_action_steps, 45)
            self.assertEqual(len(samples), 6)
            final_chunk = samples[5]["action_chunk"]
            self.assertEqual(len(final_chunk), 10)

    def test_recorded_sampler_rejects_target_without_applied_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            records = actions_path.read_text(encoding="utf-8").splitlines()
            actions_path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "complete applied-action prefix"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_remapped_action_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            records[1]["chunk_index"] = 1
            records[2]["chunk_index"] = 1
            actions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match its action index"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_joint_target_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            records[1]["joint_positions"][0] += 0.1
            actions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "joint target"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_missing_applied_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            records[2].pop("simulation_timing")
            actions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires simulation timing"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_unknown_record_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            with actions_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"schema_version": 9, "record_type": "untrusted_extension"}
                    )
                    + "\n"
                )

            with self.assertRaisesRegex(ValueError, "unknown record type"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_float_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            config_path = source / "config.json"
            config = json.loads(config_path.read_text())
            config["schema_version"] = 9.0
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema-v9"):
                _RecordedPi0Replay.load(source)


if __name__ == "__main__":
    unittest.main()
