from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import traceback
import unittest
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, cast
from unittest.mock import patch

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


def _png_base64(
    value: int,
    *,
    width: int = 640,
    height: int = 360,
    flat: bool = False,
) -> str:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    if not flat:
        pixels[:, width // 2 :] = min(value + 20, 255)
    image = Image.fromarray(pixels, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _white_sliver_png_base64(*, width: int = 640, height: int = 360) -> str:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[height // 2 - 1 : height // 2 + 1, :] = 20
    image = Image.fromarray(pixels, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _pi0_policy_profile() -> dict[str, Any]:
    return {
        "base_model": "pi0-droid",
        "openpi_config": "pi0_droid_jointpos_polaris",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris",
        "openpi_source_commit": "714ec9aa5e4e9b73b98c6bf3a328f377268e26f9",
        "action_space": "droid_joint_position",
        "action_horizon": 10,
        "action_dim": 8,
    }


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
        camera_payloads: list[str] | None = None,
        supports_camera_contract: bool = True,
        supports_active_camera: bool = True,
        transient_tool_failures: Mapping[str, int] | None = None,
        transport_tool_failures: Mapping[str, int] | None = None,
        step_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.readiness_failures = readiness_failures
        self.camera_capture_failures = camera_capture_failures
        self.robot_loaded = warm
        self.camera_payloads = list(camera_payloads or [])
        self.supports_camera_contract = supports_camera_contract
        self.supports_active_camera = supports_active_camera
        self.transient_tool_failures = dict(transient_tool_failures or {})
        self.transport_tool_failures = dict(transport_tool_failures or {})
        self.step_payloads = list(step_payloads or [])
        self.physics_dt = 1.0 / 60.0
        self.current_time = 0.0
        self.timeline_state = "stopped"
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
        transient_failures = self.transient_tool_failures.get(name, 0)
        if transient_failures > 0:
            self.transient_tool_failures[name] = transient_failures - 1
            raise RuntimeError(
                "ISAAC_UNREACHABLE: Isaac Sim MCP extension is not ready yet"
            )
        transport_failures = self.transport_tool_failures.get(name, 0)
        if transport_failures > 0:
            self.transport_tool_failures[name] = transport_failures - 1
            raise RuntimeError(f"MCP tool {name!r} failed with HTTP 502")
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
        if name == "isaac.list_prims":
            root = str(arguments["root_path"]).rstrip("/")
            prefix = f"{root}/"
            prims = []
            for path in sorted(self.prims):
                if not path.startswith(prefix):
                    continue
                relative = path[len(prefix) :]
                first = relative.split("/", 1)[0]
                child_path = f"{prefix}{first}"
                if not any(item["path"] == child_path for item in prims):
                    prims.append({"path": child_path, "type": "Xform"})
            return {"status": "success", "prims": prims}
        if name == "isaac.load_usd":
            self.robot_loaded = True
            self.prims.add(arguments["prim_path"])
            return {"status": "success"}
        if name == "isaac.delete_object":
            path = str(arguments["prim_path"])
            self.prims = {
                prim
                for prim in self.prims
                if prim != path and not prim.startswith(f"{path}/")
            }
            return {"status": "success"}
        if name == "isaac.create_camera":
            if "orientation" in arguments and not self.supports_camera_contract:
                return {"status": "error", "message": "unsupported camera contract"}
            self.prims.add(arguments["prim_path"])
            return {"status": "success"}
        if name == "isaac.set_active_camera":
            if not self.supports_active_camera:
                return {"status": "error", "message": "unsupported active camera"}
            return {
                "status": "success",
                "active_camera": arguments["prim_path"],
            }
        if name == "isaac.execute_script":
            return {"status": "success", "result": {"status": "success"}}
        if name == "isaac.step_simulation":
            if self.step_payloads:
                result = {"status": "success", **self.step_payloads.pop(0)}
            else:
                result = {"status": "success", "stepped": arguments["num_steps"]}
            self.current_time += int(result.get("stepped", 0)) * self.physics_dt
            return result
        if name == "isaac.get_simulation_state":
            return {
                "status": "success",
                "timeline_state": self.timeline_state,
                "current_time": self.current_time,
                "physics_dt": self.physics_dt,
            }
        if name == "isaac.play_simulation":
            self.timeline_state = "playing"
            return {"status": "success"}
        if name == "isaac.pause_simulation":
            self.timeline_state = "paused"
            return {"status": "success"}
        if name == "isaac.set_joint_positions":
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
                "data": (
                    self.camera_payloads.pop(0)
                    if self.camera_payloads
                    else _png_base64(20 + camera_index)
                ),
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
    def __init__(
        self,
        mcp: FakeMCP,
        *,
        stop_error: Exception | None = None,
    ) -> None:
        self.mcp = mcp
        self.stop_error = stop_error
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
        if self.stop_error is not None:
            raise self.stop_error


class FakeSampler:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        response: Any | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.observations: list[DroidObservation] = []
        self.timeouts: list[float | None] = []
        self.reset_calls = 0
        self.closed = False
        self.error = error
        self.response = response
        self.close_error = close_error

    def reset_sampling_session(self) -> None:
        self.reset_calls += 1

    def sample_droid(
        self, observation: DroidObservation, *, timeout: float | None = None
    ) -> Any:
        self.observations.append(observation)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
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
        if self.close_error is not None:
            raise self.close_error


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
        self.assertEqual(observation.exterior_image_1_left.shape, (360, 640, 3))
        self.assertEqual(int(observation.exterior_image_1_left[0, 0, 0]), 20)
        self.assertEqual(int(observation.exterior_image_1_left[0, -1, 0]), 40)
        self.assertEqual(int(observation.exterior_image_2_left[0, 0, 0]), 21)
        self.assertEqual(int(observation.wrist_image_left[0, 0, 0]), 22)

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
        self.assertEqual(step_calls[-1]["num_steps"], 4)
        self.assertAlmostEqual(result.physics_dt, 1.0 / 60.0)
        self.assertEqual(result.physics_steps_per_action, 4)
        self.assertAlmostEqual(result.control_hz, 15.0)

    def test_keeps_valid_robot_and_uses_fresh_cameras_by_default(self) -> None:
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
        self.assertEqual(len(result.created_cameras), 3)
        self.assertEqual(simulation.stopped, [])
        self.assertFalse(any(name == "isaac.load_usd" for name, _ in mcp.calls))
        self.assertEqual(
            len([name for name, _ in mcp.calls if name == "isaac.create_camera"]),
            3,
        )
        self.assertEqual(
            len([name for name, _ in mcp.calls if name == "isaac.delete_object"]),
            0,
        )

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

    def test_stop_conflict_does_not_mask_readiness_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "stop-conflict"
            simulation = FakeSimulationClient(
                FakeMCP(readiness_failures=10),
                stop_error=RuntimeError("HTTP 409: session is already failed"),
            )
            sampler = FakeSampler()
            times = iter([0.0, 1.0])
            runner = HostedDroidRunner(
                simulation,
                sampler,
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    readiness_timeout_seconds=0.5,
                    keep_session=False,
                    results_dir=results_dir,
                ),
                monotonic=lambda: next(times),
                sleep=lambda _: None,
            )

            try:
                runner.run()
            except HostedDroidError as exc:
                message = str(exc)
                traceback_functions = [
                    frame.name for frame in traceback.extract_tb(exc.__traceback__)
                ]
            else:
                self.fail("readiness failure was not raised")

            self.assertIn("Isaac MCP was not ready", message)
            self.assertEqual(traceback_functions[-1], "_wait_for_isaac")
            self.assertNotIn("stop_session", traceback_functions)
            self.assertTrue(sampler.closed)
            self.assertEqual(simulation.stopped, ["sess_hosted_droid"])
            payload = json.loads((results_dir / "error.json").read_text())
            self.assertEqual(payload["error"]["type"], "HostedDroidError")
            self.assertIn("Isaac MCP was not ready", payload["error"]["message"])
            self.assertEqual(
                payload["evidence_errors"],
                [
                    "session stop failed: RuntimeError: "
                    "HTTP 409: session is already failed"
                ],
            )

    def test_failed_run_still_stops_session_after_sampler_close_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "close-failure"
            simulation = FakeSimulationClient(FakeMCP(readiness_failures=10))
            sampler = FakeSampler(close_error=RuntimeError("close failed"))
            times = iter([0.0, 1.0])
            runner = HostedDroidRunner(
                simulation,
                sampler,
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    readiness_timeout_seconds=0.5,
                    keep_session=False,
                    results_dir=results_dir,
                ),
                monotonic=lambda: next(times),
                sleep=lambda _: None,
            )

            with self.assertRaisesRegex(HostedDroidError, "Isaac MCP was not ready"):
                runner.run()

            self.assertTrue(sampler.closed)
            self.assertEqual(simulation.stopped, ["sess_hosted_droid"])
            payload = json.loads((results_dir / "error.json").read_text())
            self.assertEqual(
                payload["evidence_errors"],
                ["sampling API close failed: RuntimeError: close failed"],
            )

    def test_configuration_rejects_invalid_rollout_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment_uri"):
            HostedDroidConfig(environment_uri="")
        with self.assertRaisesRegex(ValueError, "open_loop_horizon"):
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                open_loop_horizon=0,
            )
        first = HostedDroidConfig(environment_uri="cybernetics://envs/env_droid")
        second = HostedDroidConfig(environment_uri="cybernetics://envs/env_droid")
        self.assertTrue(
            {camera.prim_path for camera in first.cameras}.isdisjoint(
                camera.prim_path for camera in second.cameras
            )
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
            self.assertEqual(config_payload["schema_version"], 5)
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
            self.assertEqual(
                result_payload["evidence"]["actions"],
                {"path": "actions.jsonl", "records": 8},
            )
            action_records = [
                json.loads(line)
                for line in (results_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["record_type"] for record in action_records],
                [
                    "sample",
                    "action_target",
                    "applied_action",
                    "action_target",
                    "applied_action",
                    "sample",
                    "action_target",
                    "applied_action",
                ],
            )
            first_sample = action_records[0]
            self.assertEqual(first_sample["sampled_action_chunk_shape"], [2, 8])
            self.assertEqual(len(first_sample["sampled_action_chunk"]), 2)
            self.assertEqual(len(first_sample["action_chunk"]), 2)
            second_target = action_records[3]
            self.assertEqual(second_target["policy_action"][7], 1.0)
            self.assertAlmostEqual(
                second_target["joint_positions"][-1],
                GRIPPER_CLOSED_RADIANS,
            )
            self.assertEqual(len(second_target["joint_indices"]), 8)
            self.assertEqual(
                (results_dir / "actions.jsonl").stat().st_mode & 0o777,
                0o600,
            )
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

    def test_archives_predicted_video_and_sde_trajectory_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "policy-artifacts"
            response = {
                "action_chunk": np.zeros((1, 1, 8), dtype=np.float32),
                "predicted_video": np.arange(24, dtype=np.float32).reshape(
                    1, 2, 3, 2, 2
                ),
                "trajectory": [
                    {
                        "log_prob_old": np.asarray([-1.25], dtype=np.float32),
                        "log/prob": np.asarray([1.0], dtype=np.float32),
                        "log_prob": np.asarray([2.0], dtype=np.float32),
                        "step_index": np.asarray([0], dtype=np.int64),
                    }
                ],
            }
            runner = HostedDroidRunner(
                FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
                FakeSampler(response=response),
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                    policy_mode="sde",
                    include_predicted_video=True,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            runner.run()

            records = [
                json.loads(line)
                for line in (results_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 3)
            sample = records[0]
            self.assertEqual(sample["record_type"], "sample")
            self.assertEqual(sample["predicted_video"]["shape"], [1, 2, 3, 2, 2])
            video = np.load(results_dir / sample["predicted_video"]["path"])
            np.testing.assert_array_equal(video, response["predicted_video"])
            trajectory_path = results_dir / sample["trajectory"]["path"]
            with np.load(trajectory_path) as trajectory:
                step_metadata = sample["trajectory"]["steps"][0]
                np.testing.assert_array_equal(
                    trajectory[step_metadata["log_prob_old"]["archive_key"]],
                    [-1.25],
                )
                np.testing.assert_array_equal(
                    trajectory[step_metadata["step_index"]["archive_key"]],
                    [0],
                )
                colliding = step_metadata
                slash_key = colliding["log/prob"]["archive_key"]
                underscore_key = colliding["log_prob"]["archive_key"]
                self.assertNotEqual(slash_key, underscore_key)
                np.testing.assert_array_equal(trajectory[slash_key], [1.0])
                np.testing.assert_array_equal(trajectory[underscore_key], [2.0])

    def test_records_post_action_frames_as_rollout_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "video-results"
            written: dict[str, Any] = {}

            def write_video(path, frames, *, fps, codec):
                written["frames"] = list(frames)
                written["fps"] = fps
                written["codec"] = codec
                Path(path).write_bytes(b"validated-mp4")

            def read_video(_path):
                return np.stack(written["frames"])

            runner = HostedDroidRunner(
                FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
                FakeSampler(
                    response={
                        "action_chunk": np.zeros((1, 10, 8), dtype=np.float32),
                        "policy_metadata": _pi0_policy_profile(),
                    }
                ),
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    base_model="pi0-droid",
                    max_action_steps=3,
                    open_loop_horizon=3,
                    record_video=True,
                    video_fps=12,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            with patch.dict(
                "sys.modules",
                {
                    "mediapy": SimpleNamespace(
                        write_video=write_video,
                        read_video=read_video,
                    )
                },
            ):
                runner.run()

            self.assertEqual(len(written["frames"]), 3)
            self.assertEqual(written["fps"], 12)
            self.assertEqual(written["codec"], "h264")
            self.assertEqual(
                (results_dir / "rollout.mp4").read_bytes(), b"validated-mp4"
            )
            self.assertEqual(
                len(list((results_dir / "video-frames").glob("action-*.png"))),
                3,
            )
            result_payload = json.loads(
                (results_dir / "result.json").read_text(encoding="utf-8")
            )
            video = result_payload["evidence"]["video"]
            self.assertEqual(video["path"], "rollout.mp4")
            self.assertEqual(video["frames"], 3)
            self.assertEqual(video["fps"], 12)
            self.assertEqual(video["codec"], "h264")
            self.assertEqual(video["width"], 640)
            self.assertEqual(video["height"], 360)
            self.assertEqual(
                video["sha256"], hashlib.sha256(b"validated-mp4").hexdigest()
            )
            manifest = json.loads(
                (results_dir / video["source_frames_manifest"]).read_text()
            )
            self.assertEqual(len(manifest["frames"]), 3)

    def test_video_failure_does_not_mask_original_rollout_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "video-error"
            runner = HostedDroidRunner(
                FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
                FakeSampler(error=RuntimeError("policy failed")),
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                    record_video=True,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            with patch(
                "sim_evals.hosted_droid._EvidenceRecorder.finalize_video",
                side_effect=RuntimeError("encoder failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "policy failed"):
                    runner.run()

            payload = json.loads((results_dir / "error.json").read_text())
            self.assertEqual(payload["error"]["message"], "policy failed")
            self.assertEqual(
                payload["evidence_errors"],
                ["video finalization failed: RuntimeError: encoder failed"],
            )

    def test_pi0_response_requires_pinned_joint_position_profile(self) -> None:
        runner = HostedDroidRunner(
            FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
            FakeSampler(
                response={"action_chunk": np.zeros((1, 10, 8), dtype=np.float32)}
            ),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                base_model="pi0-droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(HostedDroidError, "did not prove"):
            runner.run()

    def test_control_cadence_derives_eight_steps_at_120_hz(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        mcp.physics_dt = 1.0 / 120.0
        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.physics_steps_per_action, 8)
        self.assertAlmostEqual(result.control_hz, 15.0)

    def test_gripper_actions_match_reference_binary_threshold(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        actions = np.zeros((1, 3, 8), dtype=np.float32)
        actions[0, :, 7] = [0.49, 0.5, 0.51]
        HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(response={"action_chunk": actions}),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=3,
                open_loop_horizon=3,
            ),
            sleep=lambda _: None,
        ).run()

        set_calls = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.set_joint_positions"
        ]
        self.assertEqual(
            [call["joint_positions"][-1] for call in set_calls],
            [0.0, 0.0, GRIPPER_CLOSED_RADIANS],
        )

    def test_archives_full_sampled_chunk_before_open_loop_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "full-policy-output"
            sampled_chunk = np.arange(32, dtype=np.float32).reshape(1, 4, 8)
            runner = HostedDroidRunner(
                FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
                FakeSampler(response={"action_chunk": sampled_chunk}),
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                    open_loop_horizon=2,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            runner.run()

            sample = json.loads(
                (results_dir / "actions.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(sample["sampled_action_chunk_shape"], [4, 8])
            np.testing.assert_array_equal(
                sample["sampled_action_chunk"], sampled_chunk[0]
            )
            np.testing.assert_array_equal(sample["action_chunk"], sampled_chunk[0, :2])

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

    def test_camera_transport_retry_does_not_advance_physics(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            transport_tool_failures={"isaac.capture_camera_image": 1},
        )
        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.action_steps, 1)
        retry_steps = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.step_simulation" and arguments == {"num_steps": 2}
        ]
        self.assertEqual(retry_steps, [])

    def test_retries_idempotent_joint_target_after_transport_502(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            transport_tool_failures={"isaac.set_joint_positions": 1},
        )
        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.action_steps, 1)
        joint_targets = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.set_joint_positions"
        ]
        self.assertEqual(len(joint_targets), 2)
        self.assertEqual(joint_targets[0], joint_targets[1])

    def test_partial_action_step_fails_without_applied_action_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "partial-step"
            mcp = FakeMCP(
                readiness_failures=0,
                warm=True,
                step_payloads=[
                    {"stepped": 32},
                    {"stepped": 5, "timed_out": True},
                ],
            )
            runner = HostedDroidRunner(
                FakeSimulationClient(mcp),
                FakeSampler(),
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                    physics_steps_per_action=12,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            with self.assertRaisesRegex(HostedDroidError, "incomplete action"):
                runner.run()

            records = [
                json.loads(line)
                for line in (results_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["record_type"] for record in records],
                ["sample", "action_target"],
            )
            self.assertFalse((results_dir / "result.json").exists())
            self.assertTrue((results_dir / "error.json").is_file())

    def test_jsonl_writer_retries_short_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "short-writes"
            real_write = os.write
            writes = 0

            def short_write(descriptor: int, data: bytes | memoryview) -> int:
                nonlocal writes
                writes += 1
                length = max(1, len(data) // 2)
                return real_write(descriptor, data[:length])

            with patch("sim_evals.hosted_droid.os.write", side_effect=short_write):
                HostedDroidRunner(
                    FakeSimulationClient(FakeMCP(readiness_failures=0, warm=True)),
                    FakeSampler(),
                    HostedDroidConfig(
                        environment_uri="cybernetics://envs/env_droid",
                        max_action_steps=1,
                        results_dir=results_dir,
                    ),
                    sleep=lambda _: None,
                ).run()

            records = [
                json.loads(line)
                for line in (results_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertGreater(writes, len(records))
            self.assertEqual(
                [record["record_type"] for record in records],
                ["sample", "action_target", "applied_action"],
            )

    def test_retires_previous_evaluator_camera_generations(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        mcp.prims.update(
            {
                "/World/droid_eval_old/external_cam",
                "/World/droid_eval_old/external_cam_2",
                "/World/robot/Gripper/Robotiq_2F_85/base_link/droid_eval_wrist_cam_old",
            }
        )
        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        deleted = [
            arguments["prim_path"]
            for name, arguments in mcp.calls
            if name == "isaac.delete_object"
        ]
        self.assertIn("/World/droid_eval_old", deleted)
        self.assertIn(
            "/World/robot/Gripper/Robotiq_2F_85/base_link/droid_eval_wrist_cam_old",
            deleted,
        )
        self.assertTrue(set(result.created_cameras).isdisjoint(deleted))

    def test_does_not_retry_non_idempotent_step_after_transport_502(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            transport_tool_failures={"isaac.step_simulation": 1},
        )
        runner = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(environment_uri="cybernetics://envs/env_droid"),
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(HostedDroidError, "HTTP 502"):
            runner._call(mcp, "isaac.step_simulation", {"num_steps": 1})

        step_calls = [name for name, _ in mcp.calls if name == "isaac.step_simulation"]
        self.assertEqual(len(step_calls), 1)

    def test_all_black_frames_fail_closed_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "black-frame"
            mcp = FakeMCP(
                readiness_failures=0,
                warm=True,
                camera_payloads=[_png_base64(0, flat=True)] * 10,
            )
            sampler = FakeSampler()
            runner = HostedDroidRunner(
                FakeSimulationClient(mcp),
                sampler,
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                    results_dir=results_dir,
                ),
                sleep=lambda _: None,
            )

            with self.assertRaisesRegex(HostedDroidError, "low-information frame"):
                runner.run()

            self.assertEqual(sampler.observations, [])
            self.assertFalse((results_dir / "result.json").exists())
            self.assertTrue((results_dir / "error.json").is_file())
            self.assertEqual(list((results_dir / "frames").glob("*.png")), [])

    def test_black_frames_retry_until_textured_frame_is_ready(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            camera_payloads=[
                _png_base64(0, flat=True),
                _png_base64(0, flat=True),
                _png_base64(30),
            ],
        )
        sampler = FakeSampler()

        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            sampler,
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.action_steps, 1)
        self.assertEqual(len(sampler.observations), 1)
        self.assertEqual(
            len(
                [name for name, _ in mcp.calls if name == "isaac.capture_camera_image"]
            ),
            5,
        )

    def test_near_white_geometry_sliver_fails_closed_before_sampling(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            camera_payloads=[_white_sliver_png_base64()] * 10,
        )
        sampler = FakeSampler()

        with self.assertRaisesRegex(HostedDroidError, "low-information frame"):
            HostedDroidRunner(
                FakeSimulationClient(mcp),
                sampler,
                HostedDroidConfig(
                    environment_uri="cybernetics://envs/env_droid",
                    max_action_steps=1,
                ),
                sleep=lambda _: None,
            ).run()

        self.assertEqual(sampler.observations, [])

    def test_camera_contract_includes_droid_optics_and_legacy_fallback(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            supports_camera_contract=False,
        )

        HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        enhanced = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.create_camera" and "orientation" in arguments
        ]
        scripts = [
            arguments["code"]
            for name, arguments in mcp.calls
            if name == "isaac.execute_script"
        ]
        self.assertEqual(len(enhanced), 3)
        self.assertEqual(enhanced[0]["orientation"], [-0.393, -0.195, 0.399, 0.805])
        self.assertEqual(enhanced[0]["focal_length"], 2.1)
        self.assertEqual(enhanced[0]["clipping_range"], [0.05, 100.0])
        self.assertEqual(enhanced[0]["horizontal_aperture"], 5.376)
        self.assertEqual(enhanced[2]["focal_length"], 2.8)
        self.assertEqual(len(scripts), 3)
        self.assertTrue(all("AddOrientOp" in script for script in scripts))
        self.assertTrue(all("GetClippingRangeAttr" in script for script in scripts))
        fallback_calls = [
            (name, arguments)
            for name, arguments in mcp.calls
            if name in {"isaac.create_camera", "isaac.execute_script"}
        ]
        for camera_index in range(3):
            offset = camera_index * 3
            self.assertIn("orientation", fallback_calls[offset][1])
            self.assertEqual(fallback_calls[offset + 1][0], "isaac.execute_script")
            self.assertEqual(fallback_calls[offset + 2][0], "isaac.create_camera")
            self.assertNotIn("orientation", fallback_calls[offset + 2][1])

    def test_rollout_frames_streamed_viewer_with_first_external_camera(self) -> None:
        mcp = FakeMCP(readiness_failures=0, warm=True)
        config = HostedDroidConfig(
            environment_uri="cybernetics://envs/env_droid",
            max_action_steps=1,
        )

        HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            config,
            sleep=lambda _: None,
        ).run()

        viewer_calls = [
            arguments
            for name, arguments in mcp.calls
            if name == "isaac.set_active_camera"
        ]
        self.assertEqual(len(viewer_calls), 1)
        self.assertEqual(
            viewer_calls[0]["prim_path"],
            config.cameras[0].prim_path,
        )

    def test_rollout_falls_back_to_execute_script_for_active_camera(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            supports_active_camera=False,
        )

        HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        scripts = [
            arguments["code"]
            for name, arguments in mcp.calls
            if name == "isaac.execute_script"
        ]
        self.assertEqual(len(scripts), 1)
        self.assertIn("viewport.camera_path", scripts[0])

    def test_rollout_retries_transient_bridge_readiness_failures(self) -> None:
        mcp = FakeMCP(
            readiness_failures=0,
            warm=True,
            transient_tool_failures={
                "isaac.capture_camera_image": 2,
                "isaac.step_simulation": 2,
            },
        )

        result = HostedDroidRunner(
            FakeSimulationClient(mcp),
            FakeSampler(),
            HostedDroidConfig(
                environment_uri="cybernetics://envs/env_droid",
                max_action_steps=1,
            ),
            sleep=lambda _: None,
        ).run()

        self.assertEqual(result.action_steps, 1)
        self.assertGreaterEqual(
            len(
                [name for name, _ in mcp.calls if name == "isaac.capture_camera_image"]
            ),
            5,
        )

    def test_timestamped_results_directory_is_utc_and_collision_resistant(self) -> None:
        now = datetime(2026, 7, 12, 14, 5, 6, 123456, tzinfo=timezone.utc)

        path = _timestamped_results_dir(now)

        self.assertEqual(
            path,
            Path("runs/hosted-droid/20260712T140506.123456Z"),
        )


if __name__ == "__main__":
    unittest.main()
