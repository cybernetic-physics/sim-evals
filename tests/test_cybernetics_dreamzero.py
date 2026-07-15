from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from typing import Any

import numpy as np

from sim_evals.episode_results import EpisodeResultWriter
from sim_evals.inference.cybernetics_dreamzero import (
    Client,
    CyberneticsSDKDroidSamplingAPI,
)
from sim_evals.inference.droid_observation import DroidObservation


def _observation() -> dict[str, Any]:
    return {
        "policy": {
            "external_cam": np.full((1, 4, 6, 3), 11, dtype=np.uint8),
            "external_cam_2": np.full((1, 4, 6, 3), 22, dtype=np.uint8),
            "wrist_cam": np.full((1, 4, 6, 3), 33, dtype=np.uint8),
            "arm_joint_pos": np.arange(7, dtype=np.float32),
            "gripper_pos": np.array([0.25], dtype=np.float32),
        }
    }


class FakeSamplingAPI:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[DroidObservation] = []
        self.timeouts: list[float | None] = []
        self.reset_calls = 0
        self.closed = False

    def sample_droid(
        self,
        observation: DroidObservation,
        *,
        dsrl_action: np.ndarray | None = None,
        timeout: float | None = None,
    ) -> Any:
        if dsrl_action is not None:
            raise AssertionError("non-RL client should not provide a DSRL action")
        self.requests.append(observation)
        self.timeouts.append(timeout)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    def reset_sampling_session(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


class CyberneticsDreamZeroClientTest(unittest.TestCase):
    def test_concrete_bridge_uses_typed_sdk_sample_and_cancels_session(self) -> None:
        sampled: list[object] = []
        sampled_options: list[dict[str, object]] = []
        cancelled: list[str] = []
        closed: list[bool] = []

        class SDKObservation:
            @classmethod
            def from_numpy(cls, **kwargs):
                return kwargs

        class Future:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                return self.value

        class Sampler:
            def sample_droid(self, observation, **kwargs):
                sampled.append(observation)
                sampled_options.append(kwargs)
                return Future({"action_chunk": np.zeros((1, 1, 8))})

        class RestClient:
            def cancel_session(self, session_id):
                cancelled.append(session_id)
                return Future({"ok": True})

        class ServiceClient:
            session_id = "session-1"
            holder = SimpleNamespace(close=lambda: closed.append(True))

            def create_sampling_client(self, **_kwargs):
                return Sampler()

            def create_rest_client(self):
                return RestClient()

        fake_sdk = SimpleNamespace(
            DroidObservation=SDKObservation,
            ServiceClient=ServiceClient,
        )
        observation = DroidObservation.from_sim_observation(
            _observation(), "put the cube in the bowl"
        )
        with patch.dict("sys.modules", {"cybernetics": fake_sdk}):
            api = CyberneticsSDKDroidSamplingAPI(
                policy_mode="sde", include_predicted_video=True
            )
            api.reset_sampling_session()
            response = api.sample_droid(observation, timeout=19)
            api.close()

        self.assertEqual(response["action_chunk"].shape, (1, 1, 8))
        self.assertEqual(sampled[0]["instruction"], "put the cube in the bowl")
        self.assertEqual(
            sampled_options,
            [{"policy_mode": "sde", "include_predicted_video": True}],
        )
        self.assertEqual(cancelled, ["session-1"])
        self.assertEqual(closed, [True])

    def test_pi0_dsrl_bridge_requires_matching_noise_acknowledgement(self) -> None:
        observed_actions: list[np.ndarray] = []
        checked_metadata: list[object] = []

        class SDKObservation:
            @classmethod
            def from_numpy(cls, **kwargs):
                return kwargs

        class SDKDsrlAction:
            @classmethod
            def from_numpy(cls, values):
                observed_actions.append(np.asarray(values).copy())
                return cls()

            def require_applied_policy_metadata(self, metadata):
                checked_metadata.append(metadata)
                if metadata.get("ack") != "matching":
                    raise ValueError("noise acknowledgement mismatch")

        class Future:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                return self.value

        class Sampler:
            metadata = {"ack": "matching"}

            def sample_droid(self, _observation, **kwargs):
                self.options = kwargs
                return Future(
                    {
                        "action_chunk": np.zeros((10, 8), dtype=np.float32),
                        "policy_metadata": dict(self.metadata),
                    }
                )

        sampler = Sampler()

        class ServiceClient:
            session_id = "session-1"
            holder = SimpleNamespace(close=lambda: None)

            def create_sampling_client(self, **_kwargs):
                return sampler

            def create_rest_client(self):
                return SimpleNamespace(
                    cancel_session=lambda _session_id: Future({"ok": True})
                )

        fake_sdk = SimpleNamespace(
            DroidObservation=SDKObservation,
            Pi0DroidDsrlAction=SDKDsrlAction,
            ServiceClient=ServiceClient,
        )
        observation = DroidObservation.from_sim_observation(
            _observation(), "put the cube in the bowl"
        )
        action = np.arange(32, dtype=np.float32)
        with patch.dict("sys.modules", {"cybernetics": fake_sdk}):
            api = CyberneticsSDKDroidSamplingAPI(base_model="pi0-droid")
            response = api.sample_droid(observation, dsrl_action=action, timeout=19)
            sampler.metadata = {"ack": "different"}
            with self.assertRaisesRegex(ValueError, "noise acknowledgement mismatch"):
                api.sample_droid(observation, dsrl_action=action, timeout=19)
            api.close()

        np.testing.assert_array_equal(observed_actions[0], action)
        self.assertIs(sampler.options["dsrl_action"].__class__, SDKDsrlAction)
        self.assertEqual(
            checked_metadata,
            [{"ack": "matching"}, {"ack": "different"}],
        )
        self.assertEqual(response["action_chunk"].shape, (10, 8))

    def test_collects_raw_droid_observation_and_consumes_action_chunk(self) -> None:
        action_chunk = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
        api = FakeSamplingAPI([{"action_chunk": action_chunk}])
        client = Client(sampling_api=api, request_timeout=19.0)

        client.reset()
        first = client.infer(_observation(), "put the cube in the bowl")
        second = client.infer(_observation(), "put the cube in the bowl")

        self.assertEqual(api.reset_calls, 1)
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.timeouts, [19.0])
        request = api.requests[0]
        np.testing.assert_array_equal(request.exterior_image_1_left, 11)
        np.testing.assert_array_equal(request.exterior_image_2_left, 22)
        np.testing.assert_array_equal(request.wrist_image_left, 33)
        np.testing.assert_array_equal(request.joint_position, np.arange(7))
        np.testing.assert_array_equal(request.gripper_position, [0.25])
        self.assertEqual(request.instruction, "put the cube in the bowl")
        np.testing.assert_array_equal(first["action"], action_chunk[0, 0])
        np.testing.assert_array_equal(second["action"], action_chunk[0, 1])
        self.assertTrue(first["sampled_new_chunk"])
        self.assertFalse(second["sampled_new_chunk"])
        self.assertEqual(first["viz"].shape, (224, 3 * 224, 3))
        self.assertEqual(client.episode_metrics()["sampling_requests"], 1)

    def test_accepts_typed_and_raw_tensor_action_chunks(self) -> None:
        class TypedTensor:
            def to_numpy(self) -> np.ndarray:
                return np.ones((1, 1, 8), dtype=np.float32)

        class TypedResponse:
            action_chunk = TypedTensor()

        responses = [
            TypedResponse(),
            {"action_chunk": {"data": list(range(8)), "shape": [1, 1, 8]}},
        ]
        api = FakeSamplingAPI(responses)
        client = Client(sampling_api=api)
        client.reset()

        typed = client.infer(_observation(), "put the cube in the bowl")
        raw = client.infer(_observation(), "put the cube in the bowl")

        np.testing.assert_array_equal(typed["action"], np.ones(8))
        np.testing.assert_array_equal(raw["action"], np.arange(8))

    def test_rejects_non_joint_position_action_chunks(self) -> None:
        api = FakeSamplingAPI([{"action_chunk": np.zeros((2, 7), dtype=np.float32)}])
        client = Client(sampling_api=api)
        client.reset()

        with self.assertRaisesRegex(ValueError, r"\[N,8\]"):
            client.infer(_observation(), "put the cube in the bowl")

        metrics = client.episode_metrics()
        self.assertEqual(metrics["sampling_requests"], 1)
        self.assertEqual(metrics["sampling_errors"][0]["phase"], "sample")

    def test_reports_sampling_errors_and_resets_each_episode(self) -> None:
        api = FakeSamplingAPI([RuntimeError("backend unavailable")])
        client = Client(sampling_api=api)
        client.reset()

        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            client.infer(_observation(), "put the cube in the bowl")

        metrics = client.episode_metrics()
        self.assertEqual(metrics["sampling_errors"][0]["type"], "RuntimeError")
        self.assertGreaterEqual(metrics["sampling_latency_ms"]["total"], 0)
        client.reset()
        self.assertEqual(api.reset_calls, 2)
        self.assertEqual(client.episode_metrics()["sampling_errors"], [])

    def test_validates_seven_joint_positions_before_sampling(self) -> None:
        observation = _observation()
        observation["policy"]["arm_joint_pos"] = np.zeros(6, dtype=np.float32)
        api = FakeSamplingAPI([])
        client = Client(sampling_api=api)

        with self.assertRaisesRegex(ValueError, "7 values"):
            client.infer(observation, "put the cube in the bowl")
        self.assertEqual(api.requests, [])

    def test_close_delegates_to_sampling_api(self) -> None:
        api = FakeSamplingAPI([])
        client = Client(sampling_api=api)
        client.close()
        self.assertTrue(api.closed)


class EpisodeResultWriterTest(unittest.TestCase):
    def test_writes_matching_jsonl_and_json_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeResultWriter(Path(directory))
            writer.record({"episode": 1, "status": "completed"})
            writer.record({"episode": 2, "status": "error"})

            lines = writer.jsonl_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"episode": 1', lines[0])
            self.assertIn('"episode": 2', writer.json_path.read_text())


if __name__ == "__main__":
    unittest.main()
