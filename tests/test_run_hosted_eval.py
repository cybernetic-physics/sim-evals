from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from run_hosted_eval import (
    _RecordedPi0Replay,
    _parser,
    _validate_replay_configuration,
)
from sim_evals.hosted_droid import (
    HostedDroidConfig,
    HostedDroidError,
    _config_dict,
    finalize_hosted_evidence_manifest,
    scene1_cube_in_bowl_success_spec,
)


_KNOWN_ARTIFACT_PRODUCER = {
    "status": "known",
    "sim_evals_revision": "a" * 40,
    "sim_evals_version": "0.1.0",
    "cybernetics_sdk_version": "0.18.0",
}


def _task_state() -> dict[str, object]:
    return {
        "object_bounds": {
            "center": [0.36, -0.08, 0.074],
            "minimum": [0.33, -0.11, 0.045],
            "maximum": [0.39, -0.05, 0.103],
            "size": [0.06, 0.06, 0.058],
        },
        "receptacle_bounds": {
            "center": [0.405, 0.174, 0.074],
            "minimum": [0.325, 0.093, 0.046],
            "maximum": [0.486, 0.255, 0.101],
            "size": [0.161, 0.162, 0.055],
        },
        "velocity_source": "physics_tensor",
        "object_linear_velocity": [0.0, 0.0, 0.0],
        "object_angular_velocity": [0.0, 0.0, 0.0],
        "receptacle_linear_velocity": [0.0, 0.0, 0.0],
        "receptacle_angular_velocity": [0.0, 0.0, 0.0],
        "object_runtime_position": [0.36, -0.08, 0.074],
        "gripper_reference_position": [0.36, 0.0, 0.472],
    }


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
    source_config = HostedDroidConfig(
        environment_uri="cybernetics://envs/env_droid",
        base_model="pi0-droid",
        instruction="put the cube in the bowl",
        max_action_steps=applied_action_steps,
        open_loop_horizon=8,
        physics_hz=120.0,
        task_success=scene1_cube_in_bowl_success_spec(),
    )
    config = {
        "schema_version": 9,
        "config": _config_dict(source_config),
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
    (directory / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 9,
                "base_model": "pi0-droid",
                "action_source": "worldlines_policy",
                "physics_dt": 1.0 / 120.0,
                "physics_steps_per_action": 8,
                "physics_hz": 120.0,
                "target_control_hz": 15.0,
                "control_hz": 15.0,
                "solver_position_iterations": 64,
                "solver_velocity_iterations": 1,
                "initial_arm_joint_positions": [0.0] * 7,
                "initial_gripper_position": 0.0,
                "task_success_predicate": "scene1-cube-in-bowl",
            }
        ),
        encoding="utf-8",
    )
    task_records = [
        {
            "schema_version": 9,
            "record_type": "task_state",
            "action_index": action_index,
            "phase": "initial" if action_index is None else "post_action",
            "state": _task_state(),
            "evaluation": {"predicate": "scene1-cube-in-bowl"},
        }
        for action_index in [None, *range(applied_action_steps)]
    ]
    (directory / "task-states.jsonl").write_text(
        "\n".join(json.dumps(record) for record in task_records) + "\n",
        encoding="utf-8",
    )
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 9,
                "status": "succeeded",
                "execution_status": "completed",
                "task_status": "failed",
                "result": {
                    "action_steps": applied_action_steps,
                    "task_success": False,
                    "task_success_predicate": "scene1-cube-in-bowl",
                },
            }
        ),
        encoding="utf-8",
    )
    finalize_hosted_evidence_manifest(
        directory,
        terminal_record="result.json",
        artifact_producer=_KNOWN_ARTIFACT_PRODUCER,
    )


def _refresh_manifest(directory: Path) -> None:
    finalize_hosted_evidence_manifest(
        directory,
        terminal_record="result.json",
        artifact_producer=_KNOWN_ARTIFACT_PRODUCER,
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
            self.assertEqual(metadata["replay_actions_sha256"], sampler.actions_sha256)
            self.assertGreaterEqual(len(sampler.source_files_sha256), 5)
            self.assertEqual(
                sampler.source_artifact_producer["sim_evals_revision"],
                "a" * 40,
            )
            self.assertEqual(
                sampler.source_initial_task_state["velocity_source"],
                "physics_tensor",
            )
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                sampler.sample_droid(object(), timeout=1.0)

    def test_recorded_sampler_rejects_unmanifested_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            with (source / "actions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")

            with self.assertRaisesRegex(ValueError, "manifest.*inventory"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_unknown_artifact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            finalize_hosted_evidence_manifest(
                source,
                terminal_record="result.json",
            )

            with self.assertRaisesRegex(ValueError, "known artifact provenance"):
                _RecordedPi0Replay.load(source)

    def test_manifest_identity_binds_provenance_and_terminal_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            manifest_path = source / "evidence-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["provenance"]["artifact_producer"]["sim_evals_revision"] = "b" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                _RecordedPi0Replay.load(source)

            _refresh_manifest(source)
            manifest = json.loads(manifest_path.read_text())
            manifest["terminal_record"] = "error.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terminal record is missing"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_runtime_action_cadence_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            config_path = source / "config.json"
            config = json.loads(config_path.read_text())
            config["config"]["physics_hz"] = 240.0
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runtime_path = source / "runtime.json"
            runtime = json.loads(runtime_path.read_text())
            runtime.update(
                {
                    "physics_dt": 1.0 / 240.0,
                    "physics_steps_per_action": 16,
                    "physics_hz": 240.0,
                }
            )
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            _refresh_manifest(source)

            with self.assertRaisesRegex(ValueError, "action physics dt differs"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_inconsistent_terminal_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            result_path = source / "result.json"
            result = json.loads(result_path.read_text())
            result["execution_status"] = "failed"
            result["task_status"] = "passed"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(HostedDroidError, "statuses"):
                _refresh_manifest(source)

            result["execution_status"] = "completed"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(HostedDroidError, "statuses"):
                _refresh_manifest(source)

    def test_recorded_sampler_rejects_timing_discontinuity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=2)
            actions_path = source / "actions.jsonl"
            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            applied = [
                record
                for record in records
                if record.get("record_type") == "applied_action"
            ]
            applied[1]["simulation_timing"]["before"]["current_time"] += 0.01
            applied[1]["simulation_timing"]["after"]["current_time"] += 0.01
            actions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            _refresh_manifest(source)

            with self.assertRaisesRegex(ValueError, "timing is not continuous"):
                _RecordedPi0Replay.load(source)

    def test_recorded_sampler_rejects_excess_timeline_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            actions_path = source / "actions.jsonl"
            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            applied = next(
                record
                for record in records
                if record.get("record_type") == "applied_action"
            )
            timing = applied["simulation_timing"]
            drift = 0.003
            timing["after"]["current_time"] += drift
            timing["observed_simulation_seconds"] += drift
            timing["timeline_drift_seconds"] = drift
            actions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            _refresh_manifest(source)

            with self.assertRaisesRegex(ValueError, "exceeds the drift bound"):
                _RecordedPi0Replay.load(source)

    def test_replay_configuration_binds_environment_and_task_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_replay_evidence(source, applied_action_steps=1)
            sampler = _RecordedPi0Replay.load(source)
            matching = HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                base_model="pi0-droid",
                instruction="put the cube in the bowl",
                max_action_steps=1,
                open_loop_horizon=8,
                action_source="recorded_replay",
                replay_source_sha256=sampler.source_sha256,
                keep_session=False,
                task_success=scene1_cube_in_bowl_success_spec(),
            )
            _validate_replay_configuration(matching, sampler)

            with self.assertRaisesRegex(SystemExit, "environment_uri"):
                _validate_replay_configuration(
                    HostedDroidConfig(
                        environment_uri="cybernetics://envs/other",
                        base_model="pi0-droid",
                        instruction="put the cube in the bowl",
                        max_action_steps=1,
                        open_loop_horizon=8,
                        action_source="recorded_replay",
                        replay_source_sha256=sampler.source_sha256,
                        keep_session=False,
                        task_success=scene1_cube_in_bowl_success_spec(),
                    ),
                    sampler,
                )

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
            _refresh_manifest(source)

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
            _refresh_manifest(source)

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
            _refresh_manifest(source)

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
            _refresh_manifest(source)

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
            _refresh_manifest(source)

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
            _refresh_manifest(source)

            with self.assertRaisesRegex(ValueError, "schema-v9"):
                _RecordedPi0Replay.load(source)


if __name__ == "__main__":
    unittest.main()
