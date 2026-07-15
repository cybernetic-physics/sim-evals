from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from run_hosted_dsrl import (
    _EXPECTED_PI0_BASE_POLICY_LINEAGE,
    _build_controller,
    _parser,
    _validate_args,
    run_training,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(seconds=1)
        return result


class _FakeSampler:
    def __init__(self, identity: int) -> None:
        self.identity = identity
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeController:
    def __init__(self, *, base_policy_metadata: object | None = None) -> None:
        self.transitions = 0
        self.updates = 0
        self.checkpoint_calls: list[tuple[Path, bool]] = []
        self.base_policy_metadata = base_policy_metadata

    @property
    def gamma(self) -> float:
        return 0.99

    def metadata(self) -> dict[str, object]:
        return {
            "method": "test-dsrl",
            "base_policy_frozen": True,
            "transitions": self.transitions,
            "updates": self.updates,
            "base_policy_metadata": self.base_policy_metadata,
        }

    def select_action(
        self,
        _observation: object,
        *,
        deterministic: bool = False,
    ) -> object:
        return {"deterministic": deterministic}

    def record_transition(self, _transition: object) -> dict[str, int]:
        self.transitions += 1
        return {"transitions": self.transitions, "updates": self.updates}

    def train_after_trajectory(self, transition_count: int) -> dict[str, int]:
        self.updates += 2 * transition_count
        return {
            "transitions": self.transitions,
            "updates": self.updates,
            "trajectory_transitions": transition_count,
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        include_replay: bool = True,
    ) -> dict[str, object]:
        self.checkpoint_calls.append((path, include_replay))
        path.mkdir(parents=True, exist_ok=True)
        (path / "controller.pt").write_bytes(b"controller")
        return {
            "schema_version": 1,
            "checkpoint": "controller.pt",
            "include_replay": include_replay,
        }


class _FakeResult:
    def __init__(self, session_id: str, *, transitions: int, updates: int) -> None:
        self.session_id = session_id
        self.transitions = transitions
        self.updates = updates

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "dsrl_transitions": self.transitions,
            "dsrl_updates": self.updates,
        }


class HostedDsrlParserTests(unittest.TestCase):
    def test_defaults_are_a_single_bounded_training_episode(self) -> None:
        args = _parser().parse_args([])

        self.assertEqual(args.episodes, 1)
        self.assertEqual(args.eval_episodes, 0)
        self.assertEqual(args.max_action_steps, 200)
        self.assertFalse(args.record_video)

    def test_sparse_reward_runs_beyond_canary_require_acknowledgement(self) -> None:
        args = _parser().parse_args(
            [
                "--environment-uri",
                "cybernetics://envs/env_123/versions/ver_456",
                "--episodes",
                "2",
            ]
        )

        with self.assertRaisesRegex(ValueError, "allow-zero-success-training"):
            _validate_args(args)

    def test_requires_an_exact_immutable_environment_version(self) -> None:
        args = _parser().parse_args(["--environment-uri", "cybernetics://envs/env_123"])

        with self.assertRaisesRegex(ValueError, "immutable"):
            _validate_args(args)

    def test_total_episode_count_is_bounded(self) -> None:
        args = _parser().parse_args(
            [
                "--environment-uri",
                "cybernetics://envs/env_123/versions/ver_456",
                "--episodes",
                "1000",
                "--eval-episodes",
                "1",
                "--allow-zero-success-training",
            ]
        )

        with self.assertRaisesRegex(ValueError, "limited to 1000"):
            _validate_args(args)

    def test_resume_rejects_controller_hyperparameter_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            args = _parser().parse_args(
                [
                    "--environment-uri",
                    "cybernetics://envs/env_123/versions/ver_456",
                    "--resume",
                    str(checkpoint),
                    "--batch-size",
                    "8",
                ]
            )

            with self.assertRaisesRegex(ValueError, "come from --resume"):
                _build_controller(args)

    def test_resume_loads_checkpoint_with_optional_device_override(self) -> None:
        sentinel = object()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            args = _parser().parse_args(
                [
                    "--environment-uri",
                    "cybernetics://envs/env_123/versions/ver_456",
                    "--resume",
                    str(checkpoint),
                    "--device",
                    "cpu",
                ]
            )
            with patch(
                "run_hosted_dsrl.TorchDsrlController.load_checkpoint",
                return_value=sentinel,
            ) as load:
                result = _build_controller(args)

        self.assertIs(result, sentinel)
        load.assert_called_once_with(
            checkpoint,
            device="cpu",
            expected_base_policy_metadata=_EXPECTED_PI0_BASE_POLICY_LINEAGE,
            require_replay=True,
        )

    def test_new_controller_uses_reference_first_trajectory_and_canary_update(
        self,
    ) -> None:
        sentinel = object()
        args = _parser().parse_args(
            [
                "--environment-uri",
                "cybernetics://envs/env_123/versions/ver_456",
                "--initial-updates",
                "1",
            ]
        )
        with patch(
            "run_hosted_dsrl.TorchDsrlController",
            return_value=sentinel,
        ) as controller_type:
            result = _build_controller(args)

        self.assertIs(result, sentinel)
        config = controller_type.call_args.args[0]
        self.assertEqual(config.random_exploration_episodes, 1)
        self.assertEqual(config.initial_updates, 1)


class HostedDsrlTrainingTests(unittest.TestCase):
    def test_each_episode_gets_fresh_sampler_and_owned_hosted_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(
                results_dir,
                episodes=2,
                eval_episodes=1,
            )
            controller = _FakeController()
            samplers: list[_FakeSampler] = []
            runner_calls: list[dict[str, object]] = []
            controller_counts_during_rollout: list[tuple[int, int]] = []

            def sampler_factory() -> _FakeSampler:
                sampler = _FakeSampler(len(samplers))
                samplers.append(sampler)
                return sampler

            class FakeRunner:
                def __init__(
                    self,
                    simulation_client: object,
                    sampler: _FakeSampler,
                    config: object,
                    **kwargs: object,
                ) -> None:
                    runner_calls.append(
                        {
                            "simulation_client": simulation_client,
                            "sampler": sampler,
                            "config": config,
                            **kwargs,
                        }
                    )
                    self.sampler = sampler
                    self.kwargs = kwargs

                def run(self) -> _FakeResult:
                    self.sampler.close()
                    if self.kwargs["train_dsrl_controller"]:
                        self.kwargs["dsrl_controller"].record_transition(object())
                    controller_counts_during_rollout.append(
                        (controller.transitions, controller.updates)
                    )
                    return _FakeResult(
                        f"session-{self.sampler.identity}",
                        transitions=(1 if self.kwargs["train_dsrl_controller"] else 0),
                        updates=controller.updates,
                    )

            manifest = run_training(
                args,
                simulation_client=object(),
                controller=controller,
                sampler_factory=sampler_factory,
                runner_factory=FakeRunner,
                now=_Clock(),
            )

            records = _read_jsonl(results_dir / "episodes.jsonl")
            persisted_manifest = json.loads(
                (results_dir / "train-manifest.json").read_text()
            )

        self.assertEqual(len(samplers), 3)
        self.assertTrue(all(sampler.close_calls == 1 for sampler in samplers))
        self.assertEqual(len({id(sampler) for sampler in samplers}), 3)
        self.assertEqual(
            controller_counts_during_rollout,
            [(0, 0), (1, 2), (2, 4)],
        )
        self.assertEqual(len(runner_calls), 3)
        for index, call in enumerate(runner_calls):
            config = call["config"]
            self.assertIsNone(config.session_id)
            self.assertFalse(config.keep_session)
            self.assertEqual(config.base_model, "pi0-droid")
            self.assertEqual(config.open_loop_horizon, 10)
            self.assertIsNotNone(config.task_success)
            self.assertEqual(
                config.results_dir,
                results_dir / "episodes" / f"{index:06d}",
            )
        self.assertEqual(
            [call["train_dsrl_controller"] for call in runner_calls],
            [True, True, False],
        )
        self.assertEqual(
            [call["deterministic_dsrl"] for call in runner_calls],
            [False, False, True],
        )
        self.assertEqual(
            controller.checkpoint_calls,
            [
                (results_dir / "controller" / "latest", False),
                (results_dir / "controller" / "latest", False),
                (results_dir / "controller" / "checkpoint-000002", True),
            ],
        )
        self.assertEqual(
            [record["phase"] for record in records],
            ["train", "train", "eval"],
        )
        self.assertTrue(all(record["status"] == "succeeded" for record in records))
        self.assertEqual(records[0]["post_episode_training"]["updates_delta"], 2)
        self.assertEqual(records[1]["post_episode_training"]["updates_delta"], 2)
        self.assertEqual(records[0]["result"]["dsrl_updates_total"], 2)
        self.assertEqual(records[1]["result"]["dsrl_updates_total"], 4)
        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["completed_train_episodes"], 2)
        self.assertEqual(manifest["completed_eval_episodes"], 1)
        self.assertEqual(
            manifest["checkpoint_policy"]["lightweight_latest_role"],
            "inspection_or_evaluation_only",
        )
        self.assertTrue(
            manifest["checkpoint_policy"]["training_resume_requires_full_replay"]
        )
        self.assertEqual(persisted_manifest, manifest)

    def test_runner_failure_is_recorded_and_stops_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(results_dir, episodes=3, eval_episodes=0)
            controller = _FakeController()
            samplers: list[_FakeSampler] = []

            def sampler_factory() -> _FakeSampler:
                sampler = _FakeSampler(len(samplers))
                samplers.append(sampler)
                return sampler

            class FailingRunner:
                def __init__(
                    self,
                    _client: object,
                    sampler: _FakeSampler,
                    *_args: object,
                    **_kwargs: object,
                ) -> None:
                    self.sampler = sampler

                def run(self) -> None:
                    self.sampler.close()
                    raise RuntimeError("episode failed")

            with self.assertRaisesRegex(RuntimeError, "episode failed"):
                run_training(
                    args,
                    simulation_client=object(),
                    controller=controller,
                    sampler_factory=sampler_factory,
                    runner_factory=FailingRunner,
                    now=_Clock(),
                )

            records = _read_jsonl(results_dir / "episodes.jsonl")
            manifest = json.loads((results_dir / "train-manifest.json").read_text())

        self.assertEqual(len(samplers), 1)
        self.assertEqual(samplers[0].close_calls, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["error"]["type"], "RuntimeError")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_train_episodes"], 0)
        self.assertEqual(controller.checkpoint_calls, [])

    def test_constructor_failure_closes_the_new_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(results_dir, episodes=1, eval_episodes=0)
            sampler = _FakeSampler(0)

            def fail_constructor(*_args: object, **_kwargs: object) -> None:
                raise ValueError("bad runner")

            with self.assertRaisesRegex(ValueError, "bad runner"):
                run_training(
                    args,
                    simulation_client=object(),
                    controller=_FakeController(),
                    sampler_factory=lambda: sampler,
                    runner_factory=fail_constructor,
                    now=_Clock(),
                )

        self.assertEqual(sampler.close_calls, 1)

    def test_sampler_allocation_failure_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(results_dir, episodes=1, eval_episodes=0)

            def fail_sampler() -> None:
                raise RuntimeError("sampling capacity unavailable")

            with self.assertRaisesRegex(RuntimeError, "capacity unavailable"):
                run_training(
                    args,
                    simulation_client=object(),
                    controller=_FakeController(),
                    sampler_factory=fail_sampler,
                    now=_Clock(),
                )

            records = _read_jsonl(results_dir / "episodes.jsonl")
            manifest = json.loads((results_dir / "train-manifest.json").read_text())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(manifest["status"], "failed")

    def test_rejects_mismatched_controller_lineage_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(results_dir, episodes=1, eval_episodes=0)
            lineage = dict(_EXPECTED_PI0_BASE_POLICY_LINEAGE)
            lineage["checkpoint_uri"] = "gs://different"
            sampler_calls = 0

            def sampler_factory() -> _FakeSampler:
                nonlocal sampler_calls
                sampler_calls += 1
                return _FakeSampler(0)

            with self.assertRaisesRegex(ValueError, "pinned PI0-DROID"):
                run_training(
                    args,
                    simulation_client=object(),
                    controller=_FakeController(base_policy_metadata=lineage),
                    sampler_factory=sampler_factory,
                    now=_Clock(),
                )

        self.assertEqual(sampler_calls, 0)

    def test_retains_only_bounded_full_replay_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            args = _args(
                results_dir,
                episodes=4,
                eval_episodes=0,
                extra_args=[
                    "--checkpoint-every-episodes",
                    "1",
                    "--keep-checkpoints",
                    "2",
                ],
            )
            controller = _FakeController()
            sampler_count = 0

            def sampler_factory() -> _FakeSampler:
                nonlocal sampler_count
                sampler = _FakeSampler(sampler_count)
                sampler_count += 1
                return sampler

            class FakeRunner:
                def __init__(
                    self,
                    _client: object,
                    sampler: object,
                    _config: object,
                    **kwargs: object,
                ) -> None:
                    self.sampler = sampler
                    self.controller = kwargs["dsrl_controller"]

                def run(self) -> _FakeResult:
                    self.controller.record_transition(object())
                    self.sampler.close()
                    return _FakeResult("session", transitions=1, updates=0)

            manifest = run_training(
                args,
                simulation_client=object(),
                controller=controller,
                sampler_factory=sampler_factory,
                runner_factory=FakeRunner,
                now=_Clock(),
            )

            retained = sorted(
                path.name for path in (results_dir / "controller").glob("checkpoint-*")
            )

        self.assertEqual(retained, ["checkpoint-000003", "checkpoint-000004"])
        self.assertEqual(
            [Path(path).name for path in manifest["retained_full_checkpoints"]],
            retained,
        )

    def test_refuses_to_overwrite_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory) / "run"
            results_dir.mkdir()
            (results_dir / "unrelated.txt").write_text("preserve me")
            args = _args(results_dir, episodes=1, eval_episodes=0)

            with self.assertRaisesRegex(ValueError, "must be empty"):
                run_training(
                    args,
                    simulation_client=object(),
                    controller=_FakeController(),
                    sampler_factory=lambda: _FakeSampler(0),
                    now=_Clock(),
                )


def _args(
    results_dir: Path,
    *,
    episodes: int,
    eval_episodes: int,
    extra_args: list[str] | None = None,
) -> argparse.Namespace:
    arguments = [
        "--environment-uri",
        "cybernetics://envs/env_123/versions/ver_456",
        "--episodes",
        str(episodes),
        "--eval-episodes",
        str(eval_episodes),
        "--max-action-steps",
        "10",
        "--results-dir",
        str(results_dir),
    ]
    if episodes > 1:
        arguments.append("--allow-zero-success-training")
    if extra_args:
        arguments.extend(extra_args)
    return _parser().parse_args(arguments)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
