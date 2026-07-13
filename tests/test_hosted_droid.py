from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, cast

import numpy as np
from PIL import Image

from sim_evals.hosted_droid import (
    GRIPPER_CLOSED_RADIANS,
    HostedDroidConfig,
    HostedDroidError,
    HostedDroidRunner,
    MCPClient,
)
from sim_evals.inference.droid_observation import DroidObservation
from run_hosted_eval import _timestamped_results_dir


def _png_base64(value: int) -> str:
    image = Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class FakeMCP:
    joint_names = [
        "right_outer_knuckle_joint",
        "panda_joint4",
        "panda_joint1",
        "panda_joint7",
        "finger_joint",
        "panda_joint2",
        "panda_joint5",
        "panda_joint3",
        "panda_joint6",
    ]

    def __init__(
        self,
        *,
        readiness_failures: int = 1,
        warm: bool = False,
        camera_capture_failures: int = 0,
    ) -> None:
        self.readiness_failures = readiness_failures
        self.camera_capture_failures = camera_capture_failures
        self.robot_loaded = warm
        self.prims = (
            {
                "/World/robot",
                "/World/external_cam",
                "/World/external_cam_2",
                "/World/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
            }
            if warm
            else set()
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        if name == "isaac.get_scene_info":
            if self.readiness_failures:
                self.readiness_failures -= 1
                return {"success": False, "message": "bridge starting"}
            return {"success": True, "data": {"status": "success"}}
        if name == "isaac.get_robot_info":
            if not self.robot_loaded:
                return {"status": "error", "message": "robot missing"}
            return {"status": "success", "joint_names": list(self.joint_names)}
        if name == "isaac.get_prim_info":
            if arguments["prim_path"] not in self.prims:
                return {"status": "error", "message": "prim missing"}
            prim_type = "Camera" if "cam" in str(arguments["prim_path"]) else "Xform"
            return {
                "status": "success",
                "prim_path": arguments["prim_path"],
                "type": prim_type,
            }
        if name == "isaac.load_usd":
            self.robot_loaded = True
            self.prims.add(arguments["prim_path"])
            return {"status": "success"}
        if name == "isaac.delete_object":
            self.prims.discard(arguments["prim_path"])
            return {"status": "success"}
        if name == "isaac.create_camera":
            self.prims.add(arguments["prim_path"])
            return {"status": "success"}
        if name in {
            "isaac.play_simulation",
            "isaac.step_simulation",
            "isaac.set_joint_positions",
        }:
            return {"status": "success"}
        if name == "isaac.capture_camera_image":
            if self.camera_capture_failures:
                self.camera_capture_failures -= 1
                return {"status": "error", "message": "no rendered frame"}
            return {"status": "success", "output_path": arguments["output_path"]}
        if name == "isaac.download_artifact":
            camera_index = int(arguments["path"].removesuffix(".png").rsplit("-", 1)[1])
            return {
                "status": "success",
                "encoding": "base64",
                "data": _png_base64(20 + camera_index),
            }
        if name == "isaac.get_joint_positions":
            positions = [0.0] * len(self.joint_names)
            for index in range(1, 8):
                positions[self.joint_names.index(f"panda_joint{index}")] = index / 10
            positions[self.joint_names.index("finger_joint")] = (
                GRIPPER_CLOSED_RADIANS / 2
            )
            return {"status": "success", "joint_positions": positions}
        raise AssertionError(f"unexpected MCP tool: {name}")


class FakeSimulationClient:
    def __init__(self, mcp: FakeMCP) -> None:
        self.mcp = mcp
        self.launch_calls: list[tuple[str, dict[str, Any]]] = []
        self.stopped: list[str] = []
        self.mcp_session_ids: list[str] = []
        self.wait_calls: list[tuple[str, dict[str, Any]]] = []

    def launch(self, environment_uri: str, **kwargs: Any) -> Any:
        self.launch_calls.append((environment_uri, kwargs))
        return SimpleNamespace(session_id="sess_hosted_droid")

    def wait_for_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.wait_calls.append((session_id, kwargs))
        return {"id": session_id, "status": "running"}

    def mcp_session(self, session_id: str) -> AbstractContextManager[MCPClient]:
        @contextmanager
        def session() -> Iterator[MCPClient]:
            self.mcp_session_ids.append(session_id)
            yield self.mcp

        return cast(AbstractContextManager[MCPClient], session())

    def stop_session(self, session_id: str) -> None:
        self.stopped.append(session_id)


class FakeSampler:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.observations: list[DroidObservation] = []
        self.timeouts: list[float | None] = []
        self.reset_calls = 0
        self.closed = False
        self.error = error

    def reset_sampling_session(self) -> None:
        self.reset_calls += 1

    def sample_droid(
        self, observation: DroidObservation, *, timeout: float | None = None
    ) -> Any:
        self.observations.append(observation)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return {
            "action_chunk": np.asarray(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],
                        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0],
                    ]
                ],
                dtype=np.float32,
            )
        }

    def close(self) -> None:
        self.closed = True


class HostedDroidRunnerTest(unittest.TestCase):
    def test_repairs_scene_collects_observation_and_applies_chunk(self) -> None:
        mcp = FakeMCP()
        simulation = FakeSimulationClient(mcp)
        sampler = FakeSampler()
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid/versions/ver_1",
            max_action_steps=2,
            readiness_poll_seconds=0,
            runtime_provider="vast",
            keep_session=False,
        )

        result = HostedDroidRunner(
            simulation, sampler, config, sleep=lambda _: None
        ).run()

        self.assertEqual(result.session_id, "sess_hosted_droid")
        self.assertEqual(result.samples, 1)
        self.assertEqual(result.action_steps, 2)
        self.assertTrue(result.repaired_robot)
        self.assertFalse(result.session_retained)
        self.assertEqual(len(result.created_cameras), 3)
        self.assertEqual(simulation.mcp_session_ids, ["sess_hosted_droid"])
        self.assertEqual(simulation.stopped, ["sess_hosted_droid"])
        self.assertEqual(sampler.reset_calls, 1)
        self.assertTrue(sampler.closed)
        self.assertEqual(sampler.timeouts, [2400.0])
        self.assertEqual(simulation.launch_calls[0][1]["runtime_provider"], "vast")

        observation = sampler.observations[0]
        np.testing.assert_allclose(observation.joint_position, np.arange(1, 8) / 10)
        np.testing.assert_allclose(observation.gripper_position, [0.5])
        np.testing.assert_array_equal(observation.exterior_image_1_left, 20)
        np.testing.assert_array_equal(observation.exterior_image_2_left, 21)
        np.testing.assert_array_equal(observation.wrist_image_left, 22)

        load_calls = [args for name, args in mcp.calls if name == "isaac.load_usd"]
        self.assertEqual(load_calls[0]["prim_path"], "/World/robot")
        set_calls = [
            args for name, args in mcp.calls if name == "isaac.set_joint_positions"
        ]
        expected_indices = [2, 5, 7, 1, 6, 8, 3, 4]
        self.assertEqual(set_calls[0]["joint_indices"], expected_indices)
        self.assertEqual(set_calls[1]["joint_indices"], expected_indices)
        self.assertAlmostEqual(set_calls[0]["joint_positions"][-1], 0.0)
        self.assertAlmostEqual(
            set_calls[1]["joint_positions"][-1], GRIPPER_CLOSED_RADIANS
        )
        step_calls = [
            args for name, args in mcp.calls if name == "isaac.step_simulation"
        ]
        self.assertEqual(step_calls[-1]["num_steps"], 8)

    def test_keeps_valid_scene_and_keeps_session_by_default(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        simulation = FakeSimulationClient(mcp)
        sampler = FakeSampler()
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid",
            max_action_steps=1,
        )

        result = HostedDroidRunner(simulation, sampler, config).run()

        self.assertFalse(result.repaired_robot)
        self.assertTrue(result.session_retained)
        self.assertEqual(result.created_cameras, ())
        self.assertEqual(simulation.stopped, [])
        self.assertFalse(any(name == "isaac.load_usd" for name, _ in mcp.calls))
        self.assertFalse(any(name == "isaac.create_camera" for name, _ in mcp.calls))

    def test_resumes_caller_owned_session_without_launching_or_stopping(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        simulation = FakeSimulationClient(mcp)
        sampler = FakeSampler()
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid",
            session_id="sess_slow_cold_start",
            max_action_steps=1,
            launch_timeout_seconds=2700,
        )

        result = HostedDroidRunner(simulation, sampler, config).run()

        self.assertEqual(result.session_id, "sess_slow_cold_start")
        self.assertTrue(result.session_retained)
        self.assertEqual(simulation.launch_calls, [])
        self.assertEqual(simulation.stopped, [])
        self.assertEqual(simulation.mcp_session_ids, ["sess_slow_cold_start"])
        self.assertEqual(
            simulation.wait_calls,
            [
                (
                    "sess_slow_cold_start",
                    {
                        "timeout_seconds": 2700,
                        "poll_interval_seconds": 5.0,
                    },
                )
            ],
        )

    def test_readiness_timeout_closes_sampler_and_keeps_session(self) -> None:
        mcp = FakeMCP(readiness_failures=10)
        simulation = FakeSimulationClient(mcp)
        sampler = FakeSampler()
        times = iter([0.0, 1.0])
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid",
            readiness_timeout_seconds=0.5,
        )
        runner = HostedDroidRunner(
            simulation,
            sampler,
            config,
            monotonic=lambda: next(times),
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(HostedDroidError, "Isaac MCP was not ready"):
            runner.run()

        self.assertTrue(sampler.closed)
        self.assertEqual(simulation.stopped, [])

    def test_configuration_rejects_invalid_rollout_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment_uri"):
            HostedDroidConfig(environment_uri="")
        with self.assertRaisesRegex(ValueError, "open_loop_horizon"):
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                open_loop_horizon=0,
            )

    def test_writes_config_result_and_first_rgb_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "selected-results"
            mcp = FakeMCP(readiness_failures=0, warm=True)
            simulation = FakeSimulationClient(mcp)
            sampler = FakeSampler()
            config = HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid/versions/ver_1",
                instruction="put the cube in the bowl",
                max_action_steps=3,
                open_loop_horizon=2,
                results_dir=results_dir,
            )

            result = HostedDroidRunner(simulation, sampler, config).run()

            config_payload = json.loads(
                (results_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config_payload["schema_version"], 1)
            self.assertEqual(
                config_payload["config"]["environment_uri"],
                config.environment_uri,
            )
            self.assertEqual(config_payload["config"]["results_dir"], str(results_dir))

            result_payload = json.loads(
                (results_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result_payload["status"], "succeeded")
            self.assertEqual(result_payload["result"], result.to_dict())
            self.assertEqual(len(result_payload["evidence"]["frames"]), 6)
            self.assertFalse((results_dir / "error.json").exists())

            frame_names = (
                "sample-00000-exterior-1.png",
                "sample-00000-exterior-2.png",
                "sample-00000-wrist.png",
            )
            for camera_index, frame_name in enumerate(frame_names):
                expected = base64.b64decode(_png_base64(20 + camera_index))
                self.assertEqual(
                    (results_dir / "frames" / frame_name).read_bytes(), expected
                )
            self.assertTrue(
                (results_dir / "frames" / "sample-00001-exterior-1.png").is_file()
            )

    def test_writes_structured_error_after_frames_are_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "failed-results"
            mcp = FakeMCP(readiness_failures=0, warm=True)
            simulation = FakeSimulationClient(mcp)
            sampler = FakeSampler(error=RuntimeError("sampler unavailable"))
            config = HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                results_dir=results_dir,
            )

            with self.assertRaisesRegex(RuntimeError, "sampler unavailable"):
                HostedDroidRunner(simulation, sampler, config).run()

            error_payload = json.loads(
                (results_dir / "error.json").read_text(encoding="utf-8")
            )
            self.assertEqual(error_payload["status"], "failed")
            self.assertEqual(error_payload["session_id"], "sess_hosted_droid")
            self.assertEqual(error_payload["error"]["type"], "RuntimeError")
            self.assertEqual(error_payload["error"]["message"], "sampler unavailable")
            self.assertEqual(len(error_payload["evidence"]["frames"]), 3)
            self.assertFalse((results_dir / "result.json").exists())
            self.assertTrue(sampler.closed)
            self.assertEqual(simulation.stopped, [])

    def test_retries_camera_capture_after_render_steps(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            camera_capture_failures=2,
        )
        simulation = FakeSimulationClient(mcp)
        sampler = FakeSampler()
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid",
            max_action_steps=1,
        )

        result = HostedDroidRunner(
            simulation,
            sampler,
            config,
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.action_steps, 1)
        capture_calls = [
            name for name, _ in mcp.calls if name == "isaac.capture_camera_image"
        ]
        retry_steps = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.step_simulation" and arguments == {"num_steps": 2}
        ]
        self.assertEqual(len(capture_calls), 5)
        self.assertEqual(len(retry_steps), 2)

    def test_timestamped_results_directory_is_utc_and_collision_resistant(self) -> None:
        now = datetime(2026, 7, 12, 14, 5, 6, 123456, tzinfo=timezone.utc)

        path = _timestamped_results_dir(now)

        self.assertEqual(
            path,
            Path("runs/hosted-droid/20260712T140506.123456Z"),
        )


if __name__ == "__main__":
    unittest.main()
