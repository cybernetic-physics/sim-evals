"""Hosted Cybernetic Physics DROID rollout orchestration."""

from __future__ import annotations

import base64
import io
import json
import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
from PIL import Image

from .inference.cybernetics_dreamzero import (
    DroidObservationSamplingAPI,
    _action_chunk,
)
from .inference.droid_observation import DroidObservation


class HostedDroidError(RuntimeError):
    """A hosted DROID rollout could not satisfy its runtime contract."""


class MCPClient(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class SimulationLaunch(Protocol):
    session_id: str


class SimulationClientAPI(Protocol):
    def launch(self, environment_uri: str, **kwargs: Any) -> SimulationLaunch: ...

    def mcp_session(self, session_id: str) -> AbstractContextManager[MCPClient]: ...

    def stop_session(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class CameraSpec:
    prim_path: str
    position: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]


@dataclass(frozen=True)
class HostedDroidConfig:
    environment_uri: str
    instruction: str = "put the cube in the bowl"
    robot_prim_path: str = "/World/robot"
    robot_usd_path: str = "/data/workspace/franka_robotiq_2f_85_flattened.usd"
    cameras: tuple[CameraSpec, ...] = field(
        default_factory=lambda: (
            CameraSpec(
                "/World/external_cam",
                (0.05, 0.57, 0.66),
                (52.727, 0.019, -127.934),
            ),
            CameraSpec(
                "/World/external_cam_2",
                (0.05, -0.57, 0.66),
                (52.727, -0.019, -52.039),
            ),
            CameraSpec(
                "/World/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
                (0.011, -0.031, -0.074),
                (-108.255, -1.007, 89.892),
            ),
        )
    )
    image_width: int = 640
    image_height: int = 360
    max_action_steps: int = 450
    open_loop_horizon: int = 8
    physics_steps_per_action: int = 8
    request_timeout_seconds: float = 2400.0
    launch_timeout_seconds: float = 1200.0
    readiness_timeout_seconds: float = 600.0
    readiness_poll_seconds: float = 5.0
    keep_session: bool = False

    def __post_init__(self) -> None:
        if not self.environment_uri.strip():
            raise ValueError("environment_uri must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if len(self.cameras) != 3:
            raise ValueError("DROID requires exactly three RGB cameras")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
            ("max_action_steps", self.max_action_steps),
            ("open_loop_horizon", self.open_loop_horizon),
            ("physics_steps_per_action", self.physics_steps_per_action),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class HostedDroidRunResult:
    session_id: str
    samples: int
    action_steps: int
    repaired_robot: bool
    created_cameras: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "samples": self.samples,
            "action_steps": self.action_steps,
            "repaired_robot": self.repaired_robot,
            "created_cameras": list(self.created_cameras),
        }


ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_CLOSED_RADIANS = math.pi / 4


class HostedDroidRunner:
    """Run DreamZero on DGX Spark against a hosted Isaac MCP session."""

    def __init__(
        self,
        simulation_client: SimulationClientAPI,
        sampling_api: DroidObservationSamplingAPI,
        config: HostedDroidConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.simulation_client = simulation_client
        self.sampling_api = sampling_api
        self.config = config
        self._monotonic = monotonic
        self._sleep = sleep

    def run(self) -> HostedDroidRunResult:
        launch: SimulationLaunch | None = None
        try:
            launch = self.simulation_client.launch(
                self.config.environment_uri,
                wait=True,
                timeout_seconds=self.config.launch_timeout_seconds,
                poll_interval_seconds=self.config.readiness_poll_seconds,
            )
            mcp_session = getattr(self.simulation_client, "mcp_session", None)
            if not callable(mcp_session):
                raise HostedDroidError(
                    "Cybernetics SimulationClient must expose mcp_session(session_id)"
                )
            mcp_session = cast(
                Callable[[str], AbstractContextManager[MCPClient]], mcp_session
            )
            with mcp_session(launch.session_id) as mcp:
                self._wait_for_isaac(mcp)
                repaired_robot = self._ensure_robot(mcp)
                created_cameras = self._ensure_cameras(mcp)
                joint_indices = self._joint_indices(mcp)
                self._call(mcp, "isaac.play_simulation", {})
                self.sampling_api.reset_sampling_session()
                samples, action_steps = self._rollout(mcp, joint_indices)
            return HostedDroidRunResult(
                session_id=launch.session_id,
                samples=samples,
                action_steps=action_steps,
                repaired_robot=repaired_robot,
                created_cameras=tuple(created_cameras),
            )
        finally:
            self.sampling_api.close()
            if launch is not None and not self.config.keep_session:
                self.simulation_client.stop_session(launch.session_id)

    def _wait_for_isaac(self, mcp: MCPClient) -> None:
        deadline = self._monotonic() + self.config.readiness_timeout_seconds
        last_error = "not checked"
        while True:
            try:
                self._call(mcp, "isaac.get_scene_info", {})
                return
            except Exception as exc:
                last_error = str(exc)
            if self._monotonic() >= deadline:
                raise HostedDroidError(
                    "Isaac MCP was not ready after "
                    f"{self.config.readiness_timeout_seconds}s: {last_error}"
                )
            self._sleep(self.config.readiness_poll_seconds)

    def _ensure_robot(self, mcp: MCPClient) -> bool:
        info = self._try_call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        if info is not None and _has_droid_joints(info):
            return False

        if (
            self._try_call(
                mcp,
                "isaac.get_prim_info",
                {"prim_path": self.config.robot_prim_path},
            )
            is not None
        ):
            self._call(
                mcp,
                "isaac.delete_object",
                {"prim_path": self.config.robot_prim_path},
            )
        self._call(
            mcp,
            "isaac.load_usd",
            {
                "usd_path": self.config.robot_usd_path,
                "prim_path": self.config.robot_prim_path,
            },
        )
        self._call(mcp, "isaac.play_simulation", {})
        self._call(
            mcp,
            "isaac.step_simulation",
            {"num_steps": 1, "observe_joints": [self.config.robot_prim_path]},
        )
        repaired = self._call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        if not _has_droid_joints(repaired):
            raise HostedDroidError(
                f"loaded robot at {self.config.robot_prim_path} is not DROID-compatible"
            )
        return True

    def _ensure_cameras(self, mcp: MCPClient) -> list[str]:
        created: list[str] = []
        for camera in self.config.cameras:
            existing = self._try_call(
                mcp,
                "isaac.get_prim_info",
                {"prim_path": camera.prim_path},
            )
            if existing is not None and existing.get("type") == "Camera":
                continue
            if existing is not None:
                self._call(
                    mcp,
                    "isaac.delete_object",
                    {"prim_path": camera.prim_path},
                )
            self._call(
                mcp,
                "isaac.create_camera",
                {
                    "prim_path": camera.prim_path,
                    "position": list(camera.position),
                    "rotation": list(camera.rotation_degrees),
                    "resolution": [self.config.image_width, self.config.image_height],
                },
            )
            created.append(camera.prim_path)
        return created

    def _joint_indices(self, mcp: MCPClient) -> tuple[list[int], int]:
        info = self._call(
            mcp,
            "isaac.get_robot_info",
            {"prim_path": self.config.robot_prim_path},
        )
        names = info.get("joint_names")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise HostedDroidError("isaac.get_robot_info did not return joint_names")
        missing = [
            name for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME) if name not in names
        ]
        if missing:
            raise HostedDroidError(
                f"DROID robot is missing joints: {', '.join(missing)}"
            )
        return [names.index(name) for name in ARM_JOINT_NAMES], names.index(
            GRIPPER_JOINT_NAME
        )

    def _rollout(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
    ) -> tuple[int, int]:
        samples = 0
        action_steps = 0
        while action_steps < self.config.max_action_steps:
            observation = self._observation(mcp, joint_indices, samples)
            response = self.sampling_api.sample_droid(
                observation,
                timeout=self.config.request_timeout_seconds,
            )
            chunk = _action_chunk(response)[: self.config.open_loop_horizon]
            samples += 1
            for action in chunk:
                if action_steps >= self.config.max_action_steps:
                    break
                self._apply_action(mcp, joint_indices, action)
                action_steps += 1
        return samples, action_steps

    def _observation(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        sample_index: int,
    ) -> DroidObservation:
        images = [
            self._capture_rgb(mcp, camera.prim_path, sample_index, camera_index)
            for camera_index, camera in enumerate(self.config.cameras)
        ]
        positions_payload = self._call(
            mcp,
            "isaac.get_joint_positions",
            {"prim_path": self.config.robot_prim_path},
        )
        positions = np.asarray(
            positions_payload.get("joint_positions"), dtype=np.float32
        )
        arm_indices, gripper_index = joint_indices
        if positions.ndim != 1 or positions.size <= max(*arm_indices, gripper_index):
            raise HostedDroidError(
                "isaac.get_joint_positions returned an incomplete joint vector"
            )
        gripper = np.clip(
            positions[gripper_index] / GRIPPER_CLOSED_RADIANS,
            0.0,
            1.0,
        )
        return DroidObservation(
            exterior_image_1_left=images[0],
            exterior_image_2_left=images[1],
            wrist_image_left=images[2],
            joint_position=np.ascontiguousarray(positions[arm_indices]),
            gripper_position=np.asarray([gripper], dtype=np.float32),
            instruction=self.config.instruction,
        )

    def _capture_rgb(
        self,
        mcp: MCPClient,
        camera_prim_path: str,
        sample_index: int,
        camera_index: int,
    ) -> np.ndarray:
        output_path = (
            f"/data/workspace/media/droid-{sample_index:05d}-{camera_index}.png"
        )
        capture = self._call(
            mcp,
            "isaac.capture_camera_image",
            {"prim_path": camera_prim_path, "output_path": output_path},
        )
        encoded = _encoded_image(capture)
        if encoded is None:
            artifact_path = capture.get("output_path", output_path)
            artifact = self._call(
                mcp,
                "isaac.download_artifact",
                {"path": artifact_path},
            )
            encoded = _encoded_image(artifact)
        if encoded is None:
            raise HostedDroidError(
                f"Isaac did not return RGB bytes for camera {camera_prim_path}"
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise HostedDroidError(
                f"invalid RGB artifact for camera {camera_prim_path}: {exc}"
            ) from exc
        return np.ascontiguousarray(rgb)

    def _apply_action(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        action: np.ndarray,
    ) -> None:
        arm_indices, gripper_index = joint_indices
        gripper = float(np.clip(action[7], 0.0, 1.0) * GRIPPER_CLOSED_RADIANS)
        self._call(
            mcp,
            "isaac.set_joint_positions",
            {
                "prim_path": self.config.robot_prim_path,
                "joint_positions": [*action[:7].astype(float).tolist(), gripper],
                "joint_indices": [*arm_indices, gripper_index],
            },
        )
        self._call(
            mcp,
            "isaac.step_simulation",
            {
                "num_steps": self.config.physics_steps_per_action,
                "observe_joints": [self.config.robot_prim_path],
                "observe_cap": 1,
            },
        )

    def _try_call(
        self, mcp: MCPClient, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        try:
            return self._call(mcp, name, arguments)
        except HostedDroidError:
            return None

    def _call(
        self, mcp: MCPClient, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            raw = mcp.call_tool(name, dict(arguments))
            payload = _tool_payload(raw)
            _raise_tool_error(name, payload)
            if isinstance(payload.get("data"), Mapping):
                payload = dict(payload["data"])
                _raise_tool_error(name, payload)
            return payload
        except HostedDroidError:
            raise
        except Exception as exc:
            raise HostedDroidError(f"{name} failed: {exc}") from exc


def _tool_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, Mapping):
        return dict(structured)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
    raise HostedDroidError(
        f"MCP tool returned unsupported result type {type(result).__name__}"
    )


def _raise_tool_error(name: str, payload: Mapping[str, Any]) -> None:
    status = str(payload.get("status", "")).lower()
    if payload.get("success") is False or status in {"error", "failed", "failure"}:
        message = payload.get("message") or payload.get("error") or "unknown error"
        raise HostedDroidError(f"{name} failed: {message}")


def _has_droid_joints(payload: Mapping[str, Any]) -> bool:
    names = payload.get("joint_names")
    return isinstance(names, list) and all(
        name in names for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME)
    )


def _encoded_image(payload: Mapping[str, Any]) -> str | None:
    for key in ("image_base64", "base64", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None
