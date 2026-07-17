"""Hosted Cybernetic Physics DROID rollout orchestration."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
from PIL import Image

from .inference.cybernetics_dreamzero import (
    DroidObservationSamplingAPI,
    _action_chunk,
)
from .inference.droid_observation import DroidObservation

_TRANSIENT_MCP_FAILURE_MARKERS = (
    "BRIDGE_OFFLINE",
    "ISAAC_UNREACHABLE",
    "Isaac Sim MCP extension is not ready yet",
    "No bridge connected",
    "Temporary failure in name resolution",
    "Name or service not known",
    "All connection attempts failed",
)
_IDEMPOTENT_TRANSPORT_RETRY_TOOLS = frozenset(
    {
        "isaac.capture_camera_image",
        "isaac.get_simulation_state",
        "isaac.set_joint_positions",
    }
)
_TRANSIENT_TRANSPORT_FAILURE_MARKERS = (
    "HTTP 502",
    "MCP_TRANSPORT_CONNECT",
    "MCP_TRANSPORT_TIMEOUT",
    "MCP_TRANSPORT_ERROR",
)
_TRANSIENT_MCP_RETRIES = 12
_HOSTED_MCP_CREDENTIAL_TTL_SECONDS = 86_400
_DROID_EXTERNAL_CAMERA_ROOT_PREFIX = "/World/droid_eval_"
_DROID_WRIST_CAMERA_PREFIX = "droid_eval_wrist_cam_"
_DROID_VIEWER_CAMERA_NAME = "viewer_cam"
_DROID_POLICY_CAMERA_ROLES = ("exterior_1", "exterior_2", "wrist")
_DROID_CAMERA_ROLES = frozenset((*_DROID_POLICY_CAMERA_ROLES, "viewer"))
_ROBOT_METADATA_STDOUT_PREFIX = "DROID_ROBOT_METADATA="
_CAMERA_CALIBRATION_STDOUT_PREFIX = "DROID_CAMERA_CALIBRATION="
_VIEWER_CAMERA_RESOLUTION = (1280, 720)
_CAMERA_POSITION_TOLERANCE_METERS = 1e-5
_CAMERA_ORIENTATION_ALIGNMENT_TOLERANCE = 1e-6
_CAMERA_OPTICS_TOLERANCE = 1e-5
_CONTACT_INTEGRITY_SCHEMA_VERSION = 2
_CONTACT_MAX_CONTACTS_PER_PAIR = 64
_CONTACT_TRACE_SLOT_BUDGET = 8192
_CONTINUOUS_COLLISION_SCHEMA_VERSION = 2
_CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS = math.radians(5.0)
_CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR = 16
_CONTINUOUS_COLLISION_MAX_COLLIDERS_PER_PAIR = 32
_CONTINUOUS_COLLISION_SWEEP_SEMANTICS = (
    "rotation_safe_sensor_body_obb_backward_in_current_filter_frame"
)
_CONTINUOUS_COLLISION_ROTATION_ENVELOPE_METHOD = (
    "body_centered_symmetric_obb_with_chord_inflation"
)
_CONTINUOUS_COLLISION_TRANSLATION_SHAPE_SWEEP_SEMANTICS = (
    "current_sensor_collision_shapes_backward_through_relative_translation"
)
_CONTINUOUS_COLLISION_SAMPLING_SEMANTICS = (
    "initial_endpoint_overlap_then_pose_contact_rotation_safe_obb_and_"
    "exact_shape_sweep_after_each_update"
)
_CONTINUOUS_COLLISION_MOTION_EPSILON_METERS = 1e-12
_CONTACT_CAPTURE_SOURCE = "RigidPrim_contact_tensors_and_PhysX_scene_queries"
_CONTACT_QUATERNION_NORM_TOLERANCE = 1e-3
_LEGACY_CAMERA_CONTRACT_MARKERS = (
    "unsupported camera contract",
    "unexpected keyword",
    "unexpected argument",
    "extra inputs are not permitted",
)
_PI0_DROID_POLICY_PROFILE: dict[str, Any] = {
    "base_model": "pi0-droid",
    "openpi_config": "pi0_droid_jointpos_polaris",
    "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris",
    "openpi_source_commit": "714ec9aa5e4e9b73b98c6bf3a328f377268e26f9",
    "action_space": "droid_joint_position",
    "action_horizon": 10,
    "action_dim": 8,
}


class HostedDroidError(RuntimeError):
    """A hosted DROID rollout could not satisfy its runtime contract."""


class MCPClient(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class SimulationLaunch(Protocol):
    session_id: str


class SimulationClientAPI(Protocol):
    def launch(self, environment_uri: str, **kwargs: Any) -> SimulationLaunch: ...

    def wait_for_session(
        self,
        session_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> Mapping[str, Any]: ...

    def mcp_session(
        self,
        session_id: str,
        *,
        ttl_seconds: int,
    ) -> AbstractContextManager[MCPClient]: ...

    def stop_session(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class CameraSpec:
    role: str
    prim_path: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    focal_length: float
    clipping_range: tuple[float, float] = (0.01, 1_000_000.0)
    focus_distance: float = 28.0
    horizontal_aperture: float = 5.376
    vertical_aperture: float = 3.024
    projection: str = "perspective"
    horizontal_aperture_offset: float = 0.0
    vertical_aperture_offset: float = 0.0
    f_stop: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in _DROID_CAMERA_ROLES:
            raise ValueError(f"unsupported DROID camera role: {self.role!r}")
        if not self.prim_path.startswith("/World/"):
            raise ValueError("camera prim_path must be beneath /World")
        if self.projection != "perspective":
            raise ValueError("DROID cameras require perspective projection")
        vectors = (
            ("position", self.position, 3),
            ("orientation_wxyz", self.orientation_wxyz, 4),
            ("clipping_range", self.clipping_range, 2),
        )
        for name, values, size in vectors:
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"camera {name} must contain {size} finite values")
        if math.sqrt(sum(value * value for value in self.orientation_wxyz)) <= 1e-12:
            raise ValueError("camera orientation_wxyz must not be zero")
        near, far = self.clipping_range
        if near <= 0 or far <= near:
            raise ValueError("camera clipping_range must satisfy 0 < near < far")
        for name, value in (
            ("focal_length", self.focal_length),
            ("focus_distance", self.focus_distance),
            ("horizontal_aperture", self.horizontal_aperture),
            ("vertical_aperture", self.vertical_aperture),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"camera {name} must be positive and finite")
        for name, value in (
            ("horizontal_aperture_offset", self.horizontal_aperture_offset),
            ("vertical_aperture_offset", self.vertical_aperture_offset),
            ("f_stop", self.f_stop),
        ):
            if not math.isfinite(value):
                raise ValueError(f"camera {name} must be finite")
        if self.f_stop < 0:
            raise ValueError("camera f_stop must be non-negative")


@dataclass(frozen=True)
class DroidTaskSuccessSpec:
    """Causal geometry and hard-body acceptance contract for a DROID task."""

    name: str
    object_prim_path: str
    receptacle_prim_path: str
    left_finger_prim_path: str = "/World/robot/Gripper/Robotiq_2F_85/left_inner_finger"
    right_finger_prim_path: str = (
        "/World/robot/Gripper/Robotiq_2F_85/right_inner_finger"
    )
    gripper_reference_prim_path: str = "/World/robot/Gripper/Robotiq_2F_85/base_link"
    minimum_lift_meters: float = 0.03
    minimum_lift_checks: int = 2
    horizontal_containment_margin_meters: float = 0.002
    horizontal_containment_fraction: float = 0.65
    minimum_insertion_meters: float = 0.005
    vertical_tolerance_meters: float = 0.01
    maximum_linear_speed_mps: float = 0.10
    maximum_angular_speed_rps: float = 1.50
    maximum_object_displacement_per_action_meters: float = 0.10
    maximum_receptacle_size_change_meters: float = 0.01
    gripper_closed_threshold: float = 0.25
    gripper_released_threshold: float = 0.20
    maximum_contact_penetration_meters: float = 0.001
    maximum_contact_normal_impulse_ns: float = 0.5
    maximum_closed_support_loss_meters: float = 0.010
    minimum_bilateral_contact_fraction: float = 0.875
    maximum_bilateral_contact_gap_updates: int = 1
    maximum_bilateral_normal_dot: float = -0.8
    required_settled_checks: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("name", self.name),
            ("object_prim_path", self.object_prim_path),
            ("receptacle_prim_path", self.receptacle_prim_path),
            ("left_finger_prim_path", self.left_finger_prim_path),
            ("right_finger_prim_path", self.right_finger_prim_path),
            ("gripper_reference_prim_path", self.gripper_reference_prim_path),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name, value in (
            ("minimum_lift_meters", self.minimum_lift_meters),
            (
                "horizontal_containment_margin_meters",
                self.horizontal_containment_margin_meters,
            ),
            ("minimum_insertion_meters", self.minimum_insertion_meters),
            ("vertical_tolerance_meters", self.vertical_tolerance_meters),
            ("maximum_linear_speed_mps", self.maximum_linear_speed_mps),
            ("maximum_angular_speed_rps", self.maximum_angular_speed_rps),
            (
                "maximum_object_displacement_per_action_meters",
                self.maximum_object_displacement_per_action_meters,
            ),
            (
                "maximum_receptacle_size_change_meters",
                self.maximum_receptacle_size_change_meters,
            ),
            (
                "maximum_contact_penetration_meters",
                self.maximum_contact_penetration_meters,
            ),
            (
                "maximum_contact_normal_impulse_ns",
                self.maximum_contact_normal_impulse_ns,
            ),
            (
                "maximum_closed_support_loss_meters",
                self.maximum_closed_support_loss_meters,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.horizontal_containment_fraction <= 1:
            raise ValueError("horizontal_containment_fraction must be in (0, 1]")
        if not 0 < self.minimum_bilateral_contact_fraction <= 1:
            raise ValueError("minimum_bilateral_contact_fraction must be in (0, 1]")
        if not -1 <= self.maximum_bilateral_normal_dot <= 1:
            raise ValueError("maximum_bilateral_normal_dot must be in [-1, 1]")
        if (
            not 0
            <= self.gripper_released_threshold
            < self.gripper_closed_threshold
            <= 1
        ):
            raise ValueError(
                "gripper thresholds must satisfy 0 <= released < closed <= 1"
            )
        for name, value in (
            ("minimum_lift_checks", self.minimum_lift_checks),
            (
                "maximum_bilateral_contact_gap_updates",
                self.maximum_bilateral_contact_gap_updates,
            ),
            ("required_settled_checks", self.required_settled_checks),
        ):
            minimum = 0 if name == "maximum_bilateral_contact_gap_updates" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")


def scene1_cube_in_bowl_success_spec() -> DroidTaskSuccessSpec:
    """Return the immutable-scene-1 cube-in-bowl acceptance contract."""

    return DroidTaskSuccessSpec(
        name="scene1-cube-in-bowl",
        object_prim_path="/World/rubiks_cube",
        receptacle_prim_path="/World/_24_bowl",
    )


def _contact_integrity_request(spec: DroidTaskSuccessSpec) -> dict[str, Any]:
    """Return the exact rigid-body pairs needed to prove task integrity."""

    return {
        "max_contacts_per_pair": _CONTACT_MAX_CONTACTS_PER_PAIR,
        "limits": {
            "maximum_penetration_m": spec.maximum_contact_penetration_meters,
            "maximum_normal_impulse_ns": (spec.maximum_contact_normal_impulse_ns),
        },
        "continuous_collision": {
            "maximum_sensor_rotation_rad": (_CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS),
            "maximum_filter_rotation_rad": (_CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS),
            "max_hits_per_pair": _CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR,
        },
        "pairs": [
            {
                "label": _LEFT_FINGER_CONTACT,
                "sensor_path": spec.left_finger_prim_path,
                "filter_path": spec.object_prim_path,
            },
            {
                "label": _RIGHT_FINGER_CONTACT,
                "sensor_path": spec.right_finger_prim_path,
                "filter_path": spec.object_prim_path,
            },
            {
                "label": _RECEPTACLE_CONTACT,
                "sensor_path": spec.object_prim_path,
                "filter_path": spec.receptacle_prim_path,
            },
        ],
    }


@dataclass(frozen=True)
class HostedDroidConfig:
    environment_uri: str
    session_id: str | None = None
    base_model: str = "dreamzero-droid"
    instruction: str = "put the cube in the bowl"
    robot_prim_path: str = "/World/robot"
    robot_usd_path: str = "/data/workspace/franka_robotiq_2f_85_flattened.usd"
    cameras: tuple[CameraSpec, ...] = field(default_factory=lambda: _default_cameras())
    image_width: int = 640
    image_height: int = 360
    max_action_steps: int = 450
    open_loop_horizon: int = 8
    physics_steps_per_action: int | None = None
    target_control_hz: float = 15.0
    physics_hz: float = 240.0
    solver_position_iterations: int = 64
    solver_velocity_iterations: int = 1
    runtime_provider: str | None = None
    action_source: str = "worldlines_policy"
    replay_source_sha256: str | None = None
    policy_mode: str = "native"
    include_predicted_video: bool = False
    request_timeout_seconds: float = 2400.0
    launch_timeout_seconds: float = 1200.0
    readiness_timeout_seconds: float = 600.0
    readiness_poll_seconds: float = 5.0
    keep_session: bool = True
    record_video: bool = False
    video_fps: int = 15
    results_dir: Path | None = None
    task_success: DroidTaskSuccessSpec | None = None

    def __post_init__(self) -> None:
        if not self.environment_uri.strip():
            raise ValueError("environment_uri must not be empty")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.base_model.strip():
            raise ValueError("base_model must not be empty")
        if self.policy_mode not in {"native", "sde"}:
            raise ValueError("policy_mode must be native or sde")
        if len(self.cameras) != 3:
            raise ValueError("DROID requires exactly three RGB cameras")
        roles = tuple(camera.role for camera in self.cameras)
        if roles != _DROID_POLICY_CAMERA_ROLES:
            raise ValueError(
                "DROID cameras must be ordered as exterior_1, exterior_2, wrist"
            )
        paths = tuple(camera.prim_path for camera in self.cameras)
        if len(set(paths)) != len(paths):
            raise ValueError("DROID policy camera paths must be distinct")
        viewer_path = _viewer_camera_for(self.cameras[0]).prim_path
        if viewer_path in paths:
            raise ValueError("DROID viewer camera path must not overlap policy cameras")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
            ("max_action_steps", self.max_action_steps),
            ("open_loop_horizon", self.open_loop_horizon),
            ("video_fps", self.video_fps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            self.physics_steps_per_action is not None
            and self.physics_steps_per_action < 1
        ):
            raise ValueError("physics_steps_per_action must be at least 1")
        if not math.isfinite(self.target_control_hz) or self.target_control_hz <= 0:
            raise ValueError("target_control_hz must be positive and finite")
        if not math.isfinite(self.physics_hz) or self.physics_hz <= 0:
            raise ValueError("physics_hz must be positive and finite")
        cadence = self.physics_hz / self.target_control_hz
        if not math.isclose(cadence, round(cadence), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "physics_hz must be an integer multiple of target_control_hz"
            )
        for name, value in (
            ("solver_position_iterations", self.solver_position_iterations),
            ("solver_velocity_iterations", self.solver_velocity_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_source not in {"worldlines_policy", "recorded_replay"}:
            raise ValueError(
                "action_source must be worldlines_policy or recorded_replay"
            )
        if self.action_source == "recorded_replay":
            if self.replay_source_sha256 is None or not re.fullmatch(
                r"[0-9a-f]{64}", self.replay_source_sha256
            ):
                raise ValueError(
                    "recorded_replay requires a lowercase SHA-256 source digest"
                )
            if self.base_model != "pi0-droid":
                raise ValueError("recorded_replay requires base_model=pi0-droid")
            if self.session_id is not None:
                raise ValueError("recorded_replay requires a freshly launched session")
            if self.keep_session:
                raise ValueError(
                    "recorded_replay requires evaluator-owned session cleanup"
                )
        elif self.replay_source_sha256 is not None:
            raise ValueError("worldlines_policy must not set replay_source_sha256")


@dataclass(frozen=True)
class HostedDroidRunResult:
    session_id: str
    samples: int
    action_steps: int
    repaired_robot: bool
    created_cameras: tuple[str, ...]
    session_retained: bool
    physics_dt: float
    physics_steps_per_action: int
    control_hz: float
    task_success: bool | None = None
    task_success_predicate: str | None = None
    task_success_action_index: int | None = None
    task_success_checks: int = 0
    task_success_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "samples": self.samples,
            "action_steps": self.action_steps,
            "repaired_robot": self.repaired_robot,
            "created_cameras": list(self.created_cameras),
            "session_retained": self.session_retained,
            "physics_dt": self.physics_dt,
            "physics_steps_per_action": self.physics_steps_per_action,
            "control_hz": self.control_hz,
            "task_success": self.task_success,
            "task_success_predicate": self.task_success_predicate,
            "task_success_action_index": self.task_success_action_index,
            "task_success_checks": self.task_success_checks,
            "task_success_reason": self.task_success_reason,
        }


@dataclass(frozen=True)
class _AxisAlignedBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.minimum[0] + self.maximum[0]) / 2.0,
            (self.minimum[1] + self.maximum[1]) / 2.0,
            (self.minimum[2] + self.maximum[2]) / 2.0,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.maximum[0] - self.minimum[0],
            self.maximum[1] - self.minimum[1],
            self.maximum[2] - self.minimum[2],
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "center": list(self.center),
            "size": list(self.size),
        }


@dataclass(frozen=True)
class _DroidTaskState:
    object_bounds: _AxisAlignedBounds
    receptacle_bounds: _AxisAlignedBounds
    velocity_source: str
    object_linear_velocity: tuple[float, float, float]
    object_angular_velocity: tuple[float, float, float]
    receptacle_linear_velocity: tuple[float, float, float]
    receptacle_angular_velocity: tuple[float, float, float]
    object_runtime_position: tuple[float, float, float]
    gripper_reference_position: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_bounds": self.object_bounds.to_dict(),
            "receptacle_bounds": self.receptacle_bounds.to_dict(),
            "velocity_source": self.velocity_source,
            "object_linear_velocity": list(self.object_linear_velocity),
            "object_angular_velocity": list(self.object_angular_velocity),
            "receptacle_linear_velocity": list(self.receptacle_linear_velocity),
            "receptacle_angular_velocity": list(self.receptacle_angular_velocity),
            "object_runtime_position": list(self.object_runtime_position),
            "gripper_reference_position": list(self.gripper_reference_position),
        }


@dataclass(frozen=True)
class _RolloutOutcome:
    samples: int
    action_steps: int
    task_success: bool | None
    task_success_predicate: str | None
    task_success_action_index: int | None
    task_success_checks: int
    task_success_reason: str | None


_EVIDENCE_SCHEMA_VERSION = 9
_EVIDENCE_MANIFEST_NAME = "evidence-manifest.json"
_EVIDENCE_CAMERA_NAMES = ("exterior-1", "exterior-2", "wrist")
_TASK_STATE_STDOUT_PREFIX = "DROID_TASK_STATE="
_LEFT_FINGER_CONTACT = "left-finger-cube"
_RIGHT_FINGER_CONTACT = "right-finger-cube"
_RECEPTACLE_CONTACT = "cube-receptacle"


@dataclass(frozen=True)
class _ContactPose:
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class _ContinuousCollisionEvidence:
    previous_sensor_pose: _ContactPose
    current_sensor_pose: _ContactPose
    previous_filter_pose: _ContactPose
    current_filter_pose: _ContactPose
    previous_endpoint_contact: bool
    current_endpoint_contact: bool
    relative_translation_m: float
    sensor_rotation_rad: float
    filter_rotation_rad: float
    relative_rotation_rad: float
    rotation_envelope_inflation_m: float
    swept_collision_risk_detected: bool
    paired_hit_count: int
    exact_paired_hit_count: int
    broad_phase_only: bool


def _contact_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HostedDroidError(f"{name} must be an object")
    return value


def _contact_number(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0)
    ):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise HostedDroidError(f"{name} must be {qualifier}")
    return float(value)


def _contact_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise HostedDroidError(f"{name} is invalid")
    return value


def _contact_vector(
    value: Any,
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise HostedDroidError(f"{name} must contain {length} numbers")
    result = tuple(_contact_number(item, name) for item in value)
    return result


def _contact_string_list(
    value: Any,
    name: str,
    *,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise HostedDroidError(f"{name} is invalid")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise HostedDroidError(f"{name} is invalid")
        result.append(item)
    if len(set(result)) != len(result):
        raise HostedDroidError(f"{name} contains duplicates")
    return result


def _absolute_contact_path(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or len(value) < 2
        or not value.startswith("/")
    ):
        raise HostedDroidError(f"{name} must be an absolute prim path")
    return value


def _contact_numbers_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


def _quaternion_delta_radians(
    start_wxyz: tuple[float, float, float, float],
    end_wxyz: tuple[float, float, float, float],
) -> float:
    signed_dot = sum(
        start * end for start, end in zip(start_wxyz, end_wxyz, strict=True)
    )
    aligned_end = (
        tuple(-component for component in end_wxyz) if signed_dot < 0.0 else end_wxyz
    )
    difference_norm = math.sqrt(
        sum(
            (start - end) ** 2
            for start, end in zip(start_wxyz, aligned_end, strict=True)
        )
    )
    sum_norm = math.sqrt(
        sum(
            (start + end) ** 2
            for start, end in zip(start_wxyz, aligned_end, strict=True)
        )
    )
    return 4.0 * math.atan2(difference_norm, sum_norm)


def _quaternion_multiply(
    left_wxyz: tuple[float, float, float, float],
    right_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left_w, left_x, left_y, left_z = left_wxyz
    right_w, right_x, right_y, right_z = right_wxyz
    return (
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    )


def _quaternion_conjugate(
    value_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = value_wxyz
    return (w, -x, -y, -z)


def _quaternion_relative_orientation(
    sensor_wxyz: tuple[float, float, float, float],
    filter_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return _quaternion_multiply(
        _quaternion_conjugate(filter_wxyz),
        sensor_wxyz,
    )


def _vector_in_quaternion_frame(
    vector_world: tuple[float, float, float],
    frame_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    rotated = _quaternion_multiply(
        _quaternion_multiply(
            _quaternion_conjugate(frame_wxyz),
            (0.0, *vector_world),
        ),
        frame_wxyz,
    )
    return (rotated[1], rotated[2], rotated[3])


def _contact_pose(value: Any, name: str) -> _ContactPose:
    pose = _contact_mapping(value, name)
    if set(pose) != {"position_m", "orientation_wxyz"}:
        raise HostedDroidError(f"{name} fields are invalid")
    position = _contact_vector(
        pose.get("position_m"),
        f"{name}.position_m",
        length=3,
    )
    orientation = _contact_vector(
        pose.get("orientation_wxyz"),
        f"{name}.orientation_wxyz",
        length=4,
    )
    norm = math.sqrt(sum(component * component for component in orientation))
    if norm <= 1e-12 or abs(norm - 1.0) > _CONTACT_QUATERNION_NORM_TOLERANCE:
        raise HostedDroidError(f"{name}.orientation_wxyz must be a unit quaternion")
    normalized = tuple(component / norm for component in orientation)
    return _ContactPose(
        position_m=cast(tuple[float, float, float], position),
        orientation_wxyz=cast(
            tuple[float, float, float, float],
            normalized,
        ),
    )


def _contact_poses_match(left: _ContactPose, right: _ContactPose) -> bool:
    positions_match = all(
        _contact_numbers_close(first, second)
        for first, second in zip(left.position_m, right.position_m, strict=True)
    )
    orientations_match = any(
        all(
            _contact_numbers_close(first, sign * second)
            for first, second in zip(
                left.orientation_wxyz,
                right.orientation_wxyz,
                strict=True,
            )
        )
        for sign in (1.0, -1.0)
    )
    return positions_match and orientations_match


def _validate_collision_sweep(
    value: Any,
    *,
    name: str,
    filter_path: str,
    expected_max_hits: int,
    maximum_distance_m: float,
    require_available: bool = True,
) -> int:
    sweep = _contact_mapping(value, name)
    if set(sweep) != {
        "available",
        "max_hits",
        "captured_hit_count",
        "saturated",
        "hits",
    }:
        raise HostedDroidError(f"{name} is incomplete")
    if require_available and sweep.get("available") is not True:
        raise HostedDroidError(f"{name} is unavailable")
    if sweep.get("saturated") is not False:
        raise HostedDroidError(f"{name} is saturated")
    max_hits = _contact_integer(
        sweep.get("max_hits"),
        f"{name} maximum hits",
        minimum=1,
        maximum=_CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR,
    )
    if max_hits != expected_max_hits:
        raise HostedDroidError(f"{name} maximum hits are invalid")
    hits = sweep.get("hits")
    if not isinstance(hits, list) or len(hits) > max_hits:
        raise HostedDroidError(f"{name} hits are unbounded")
    captured_hit_count = _contact_integer(
        sweep.get("captured_hit_count"),
        f"{name} captured hit count",
        maximum=max_hits,
    )
    if captured_hit_count != len(hits):
        raise HostedDroidError(f"{name} hit count is inconsistent")

    filter_prefix = filter_path.rstrip("/") + "/"
    paired_hit_count = 0
    for hit in hits:
        hit_record = _contact_mapping(hit, f"{name} hit")
        if set(hit_record) != {"rigid_body_path", "collider_path", "distance_m"}:
            raise HostedDroidError(f"{name} hit fields are invalid")
        rigid_body_path = _absolute_contact_path(
            hit_record.get("rigid_body_path"),
            f"{name} rigid body path",
        )
        if rigid_body_path != filter_path:
            raise HostedDroidError(f"{name} hit path is invalid")
        paired_hit_count += 1
        collider_path = hit_record.get("collider_path")
        if collider_path is not None:
            collider_path = _absolute_contact_path(
                collider_path,
                f"{name} collider path",
            )
            if collider_path != filter_path and not collider_path.startswith(
                filter_prefix
            ):
                raise HostedDroidError(f"{name} collider path is invalid")
        hit_distance = _contact_number(
            hit_record.get("distance_m"),
            f"{name} hit distance",
            nonnegative=True,
        )
        if hit_distance > maximum_distance_m + max(
            1e-9,
            maximum_distance_m * 1e-6,
        ):
            raise HostedDroidError(f"{name} hit exceeds the relative motion")
    return paired_hit_count


def _validate_continuous_collision_evidence(
    value: Any,
    *,
    label: str,
    sensor_path: str,
    filter_path: str,
    manifold_contact: bool,
    expected_config: Mapping[str, Any],
    previous: _ContinuousCollisionEvidence | None,
    shared_poses: dict[tuple[str, str], _ContactPose],
) -> _ContinuousCollisionEvidence:
    evidence = _contact_mapping(
        value,
        f"contact integrity continuous collision evidence for {label}",
    )
    if set(evidence) != {
        "schema_version",
        "classification",
        "passed",
        "complete",
        "swept_collision_risk_detected",
        "tunneling_detected",
        "failure_reasons",
        "errors",
        "sensor_path",
        "filter_path",
        "previous_endpoint_contact",
        "current_endpoint_contact",
        "poses",
        "rotation_delta_radians",
        "maximum_rotation_radians",
        "relative_motion",
        "rotation_envelope",
        "sweep",
        "translation_shape_sweep",
        "translation_shape_sweep_semantics",
        "paired_hit_count",
        "exact_shape_sweep",
        "exact_paired_hit_count",
        "broad_phase_only",
        "sensor_collider_paths",
        "endpoint_evidence",
        "sweep_semantics",
        "diagnostic_errors",
    }:
        raise HostedDroidError("continuous collision evidence fields are invalid")
    schema_version = evidence.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _CONTINUOUS_COLLISION_SCHEMA_VERSION
    ):
        raise HostedDroidError("continuous collision evidence schema is unsupported")
    if evidence.get("sensor_path") != sensor_path:
        raise HostedDroidError("continuous collision sensor path is invalid")
    if evidence.get("filter_path") != filter_path:
        raise HostedDroidError("continuous collision filter path is invalid")

    collider_paths = _contact_string_list(
        evidence.get("sensor_collider_paths"),
        "continuous collision sensor collider paths",
        maximum=_CONTINUOUS_COLLISION_MAX_COLLIDERS_PER_PAIR,
    )
    if not collider_paths:
        raise HostedDroidError("continuous collision sensor collider paths are empty")
    sensor_prefix = sensor_path.rstrip("/") + "/"
    for collider_path in collider_paths:
        _absolute_contact_path(
            collider_path,
            "continuous collision sensor collider path",
        )
        if collider_path != sensor_path and not collider_path.startswith(sensor_prefix):
            raise HostedDroidError(
                "continuous collision sensor collider path is outside the sensor body"
            )
    if evidence.get("sweep_semantics") != _CONTINUOUS_COLLISION_SWEEP_SEMANTICS:
        raise HostedDroidError("continuous collision sweep semantics are invalid")
    if (
        evidence.get("translation_shape_sweep_semantics")
        != _CONTINUOUS_COLLISION_TRANSLATION_SHAPE_SWEEP_SEMANTICS
    ):
        raise HostedDroidError(
            "continuous collision translation shape sweep semantics are invalid"
        )
    diagnostic_errors = _contact_string_list(
        evidence.get("diagnostic_errors"),
        "continuous collision diagnostic errors",
        maximum=16,
    )
    if diagnostic_errors:
        raise HostedDroidError("continuous collision diagnostic evidence is incomplete")

    endpoint = _contact_mapping(
        evidence.get("endpoint_evidence"),
        "continuous collision endpoint evidence",
    )
    if set(endpoint) != {
        "previous_contact_or_overlap",
        "current_overlap",
        "current_manifold_contact",
        "current_contact_or_overlap",
    }:
        raise HostedDroidError("continuous collision endpoint evidence is incomplete")
    endpoint_values = {
        name: endpoint.get(name)
        for name in (
            "previous_contact_or_overlap",
            "current_overlap",
            "current_manifold_contact",
            "current_contact_or_overlap",
        )
    }
    if not all(type(item) is bool for item in endpoint_values.values()):
        raise HostedDroidError("continuous collision endpoint evidence is incomplete")
    previous_endpoint_contact = cast(
        bool,
        endpoint_values["previous_contact_or_overlap"],
    )
    current_overlap = cast(bool, endpoint_values["current_overlap"])
    current_manifold_contact = cast(
        bool,
        endpoint_values["current_manifold_contact"],
    )
    current_endpoint_contact = cast(
        bool,
        endpoint_values["current_contact_or_overlap"],
    )
    if current_manifold_contact is not manifold_contact:
        raise HostedDroidError("continuous collision manifold evidence is inconsistent")
    if current_endpoint_contact is not (current_overlap or current_manifold_contact):
        raise HostedDroidError("continuous collision endpoint evidence is inconsistent")
    if (
        evidence.get("previous_endpoint_contact") is not previous_endpoint_contact
        or evidence.get("current_endpoint_contact") is not current_endpoint_contact
    ):
        raise HostedDroidError(
            "continuous collision endpoint classification is inconsistent"
        )

    poses = _contact_mapping(evidence.get("poses"), "continuous collision poses")
    expected_pose_names = {
        "previous_sensor",
        "current_sensor",
        "previous_filter",
        "current_filter",
    }
    if set(poses) != expected_pose_names:
        raise HostedDroidError("continuous collision poses are incomplete")
    parsed_poses = {
        name: _contact_pose(poses.get(name), f"continuous collision pose {name}")
        for name in expected_pose_names
    }
    previous_sensor_pose = parsed_poses["previous_sensor"]
    current_sensor_pose = parsed_poses["current_sensor"]
    previous_filter_pose = parsed_poses["previous_filter"]
    current_filter_pose = parsed_poses["current_filter"]
    if previous is not None:
        discontinuities: list[str] = []
        if previous_endpoint_contact is not previous.current_endpoint_contact:
            discontinuities.append("endpoint_contact")
        if not _contact_poses_match(
            previous_sensor_pose,
            previous.current_sensor_pose,
        ):
            discontinuities.append("sensor_pose")
        if not _contact_poses_match(
            previous_filter_pose,
            previous.current_filter_pose,
        ):
            discontinuities.append("filter_pose")
        if discontinuities:
            raise HostedDroidError(
                "continuous collision evidence is discontinuous between updates "
                f"for {label}: {', '.join(discontinuities)}"
            )
    for phase, path, pose in (
        ("previous", sensor_path, previous_sensor_pose),
        ("current", sensor_path, current_sensor_pose),
        ("previous", filter_path, previous_filter_pose),
        ("current", filter_path, current_filter_pose),
    ):
        existing = shared_poses.get((phase, path))
        if existing is not None and not _contact_poses_match(existing, pose):
            raise HostedDroidError(
                "continuous collision poses disagree for a shared rigid body"
            )
        shared_poses[(phase, path)] = pose

    sensor_rotation = _quaternion_delta_radians(
        previous_sensor_pose.orientation_wxyz,
        current_sensor_pose.orientation_wxyz,
    )
    filter_rotation = _quaternion_delta_radians(
        previous_filter_pose.orientation_wxyz,
        current_filter_pose.orientation_wxyz,
    )
    previous_relative_orientation = _quaternion_relative_orientation(
        previous_sensor_pose.orientation_wxyz,
        previous_filter_pose.orientation_wxyz,
    )
    current_relative_orientation = _quaternion_relative_orientation(
        current_sensor_pose.orientation_wxyz,
        current_filter_pose.orientation_wxyz,
    )
    relative_rotation = _quaternion_delta_radians(
        previous_relative_orientation,
        current_relative_orientation,
    )
    rotation_delta = _contact_mapping(
        evidence.get("rotation_delta_radians"),
        "continuous collision rotation delta",
    )
    if set(rotation_delta) != {"sensor", "filter", "relative", "maximum"}:
        raise HostedDroidError("continuous collision rotation delta is incomplete")
    reported_sensor_rotation = _contact_number(
        rotation_delta.get("sensor"),
        "continuous collision sensor rotation",
        nonnegative=True,
    )
    reported_filter_rotation = _contact_number(
        rotation_delta.get("filter"),
        "continuous collision filter rotation",
        nonnegative=True,
    )
    reported_relative_rotation = _contact_number(
        rotation_delta.get("relative"),
        "continuous collision relative rotation",
        nonnegative=True,
    )
    reported_maximum_rotation = _contact_number(
        rotation_delta.get("maximum"),
        "continuous collision maximum rotation",
        nonnegative=True,
    )
    computed_maximum_rotation = max(
        sensor_rotation,
        filter_rotation,
        relative_rotation,
    )
    rotation_mismatches = [
        f"{name} computed={computed:.12g} reported={reported:.12g}"
        for name, computed, reported in (
            ("sensor", sensor_rotation, reported_sensor_rotation),
            ("filter", filter_rotation, reported_filter_rotation),
            ("relative", relative_rotation, reported_relative_rotation),
            ("maximum", computed_maximum_rotation, reported_maximum_rotation),
        )
        if not _contact_numbers_close(reported, computed)
    ]
    if rotation_mismatches:
        raise HostedDroidError(
            "continuous collision rotation delta is inconsistent "
            f"for {label}: {', '.join(rotation_mismatches)}"
        )

    maximum_rotation = _contact_mapping(
        evidence.get("maximum_rotation_radians"),
        "continuous collision rotation limits",
    )
    if set(maximum_rotation) != {"sensor", "filter"}:
        raise HostedDroidError("continuous collision rotation limits are incomplete")
    sensor_rotation_limit = _contact_number(
        maximum_rotation.get("sensor"),
        "continuous collision sensor rotation limit",
        nonnegative=True,
    )
    filter_rotation_limit = _contact_number(
        maximum_rotation.get("filter"),
        "continuous collision filter rotation limit",
        nonnegative=True,
    )
    expected_sensor_limit = _contact_number(
        expected_config.get("maximum_sensor_rotation_rad"),
        "configured continuous collision sensor rotation limit",
        nonnegative=True,
    )
    expected_filter_limit = _contact_number(
        expected_config.get("maximum_filter_rotation_rad"),
        "configured continuous collision filter rotation limit",
        nonnegative=True,
    )
    if not _contact_numbers_close(
        sensor_rotation_limit, expected_sensor_limit
    ) or not _contact_numbers_close(filter_rotation_limit, expected_filter_limit):
        raise HostedDroidError("continuous collision rotation limits are invalid")
    if (
        sensor_rotation > expected_sensor_limit + 1e-12
        or filter_rotation > expected_filter_limit + 1e-12
    ):
        raise HostedDroidError("continuous collision rotation limit was exceeded")

    relative_motion = _contact_mapping(
        evidence.get("relative_motion"),
        "continuous collision relative motion",
    )
    if set(relative_motion) != {"translation_m", "direction_unit", "distance_m"}:
        raise HostedDroidError("continuous collision relative motion is incomplete")
    reported_translation = _contact_vector(
        relative_motion.get("translation_m"),
        "continuous collision relative translation",
        length=3,
    )
    reported_direction = _contact_vector(
        relative_motion.get("direction_unit"),
        "continuous collision relative direction",
        length=3,
    )
    reported_distance = _contact_number(
        relative_motion.get("distance_m"),
        "continuous collision relative distance",
        nonnegative=True,
    )
    previous_sensor_from_filter = tuple(
        previous_sensor_pose.position_m[axis] - previous_filter_pose.position_m[axis]
        for axis in range(3)
    )
    previous_sensor_in_filter = _vector_in_quaternion_frame(
        cast(tuple[float, float, float], previous_sensor_from_filter),
        previous_filter_pose.orientation_wxyz,
    )
    current_sensor_from_filter = tuple(
        current_sensor_pose.position_m[axis] - current_filter_pose.position_m[axis]
        for axis in range(3)
    )
    current_sensor_in_filter = _vector_in_quaternion_frame(
        cast(tuple[float, float, float], current_sensor_from_filter),
        current_filter_pose.orientation_wxyz,
    )
    expected_translation = tuple(
        current_sensor_in_filter[axis] - previous_sensor_in_filter[axis]
        for axis in range(3)
    )
    expected_distance = math.sqrt(
        sum(component * component for component in expected_translation)
    )
    if expected_distance <= _CONTINUOUS_COLLISION_MOTION_EPSILON_METERS:
        expected_translation = (0.0, 0.0, 0.0)
        expected_direction = (0.0, 0.0, 0.0)
        expected_distance = 0.0
    else:
        expected_direction = tuple(
            component / expected_distance for component in expected_translation
        )
    if (
        not all(
            _contact_numbers_close(actual, expected)
            for actual, expected in zip(
                reported_translation,
                expected_translation,
                strict=True,
            )
        )
        or not all(
            _contact_numbers_close(actual, expected)
            for actual, expected in zip(
                reported_direction,
                expected_direction,
                strict=True,
            )
        )
        or not _contact_numbers_close(reported_distance, expected_distance)
    ):
        raise HostedDroidError("continuous collision relative motion is inconsistent")

    rotation_envelope = _contact_mapping(
        evidence.get("rotation_envelope"),
        "continuous collision rotation envelope",
    )
    if set(rotation_envelope) != {
        "method",
        "base_half_extents_m",
        "radius_m",
        "relative_rotation_rad",
        "inflation_m",
        "query_half_extents_m",
        "query_kind",
    }:
        raise HostedDroidError(
            "continuous collision rotation envelope fields are invalid"
        )
    if (
        rotation_envelope.get("method")
        != _CONTINUOUS_COLLISION_ROTATION_ENVELOPE_METHOD
    ):
        raise HostedDroidError(
            "continuous collision rotation envelope method is invalid"
        )
    base_half_extents = _contact_vector(
        rotation_envelope.get("base_half_extents_m"),
        "continuous collision rotation envelope base half extents",
        length=3,
    )
    if not all(component > 0 for component in base_half_extents):
        raise HostedDroidError(
            "continuous collision rotation envelope base half extents must be positive"
        )
    expected_radius = math.sqrt(
        sum(component * component for component in base_half_extents)
    )
    radius = _contact_number(
        rotation_envelope.get("radius_m"),
        "continuous collision rotation envelope radius",
        nonnegative=True,
    )
    envelope_relative_rotation = _contact_number(
        rotation_envelope.get("relative_rotation_rad"),
        "continuous collision rotation envelope relative rotation",
        nonnegative=True,
    )
    inflation = _contact_number(
        rotation_envelope.get("inflation_m"),
        "continuous collision rotation envelope inflation",
        nonnegative=True,
    )
    query_half_extents = _contact_vector(
        rotation_envelope.get("query_half_extents_m"),
        "continuous collision rotation envelope query half extents",
        length=3,
    )
    if not all(component > 0 for component in query_half_extents):
        raise HostedDroidError(
            "continuous collision rotation envelope query half extents must be positive"
        )
    expected_inflation = 2.0 * expected_radius * math.sin(relative_rotation / 2.0)
    expected_query_half_extents = tuple(
        component + expected_inflation for component in base_half_extents
    )
    expected_query_kind = (
        "sweep_box_all"
        if expected_distance > _CONTINUOUS_COLLISION_MOTION_EPSILON_METERS
        else "overlap_box"
    )
    if (
        not _contact_numbers_close(radius, expected_radius)
        or not _contact_numbers_close(
            envelope_relative_rotation,
            relative_rotation,
        )
        or not _contact_numbers_close(inflation, expected_inflation)
        or not all(
            _contact_numbers_close(actual, expected)
            for actual, expected in zip(
                query_half_extents,
                expected_query_half_extents,
                strict=True,
            )
        )
        or rotation_envelope.get("query_kind") != expected_query_kind
    ):
        raise HostedDroidError("continuous collision rotation envelope is inconsistent")

    expected_max_hits = _contact_integer(
        expected_config.get("max_hits_per_pair"),
        "configured continuous collision maximum hits",
        minimum=1,
        maximum=_CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR,
    )
    paired_hit_count = _validate_collision_sweep(
        evidence.get("sweep"),
        name="continuous collision safety sweep evidence",
        filter_path=filter_path,
        expected_max_hits=expected_max_hits,
        maximum_distance_m=expected_distance,
    )
    exact_query_required = (
        expected_distance > _CONTINUOUS_COLLISION_MOTION_EPSILON_METERS
        or paired_hit_count > 0
    )
    translation_shape_paired_hit_count = _validate_collision_sweep(
        evidence.get("translation_shape_sweep"),
        name="continuous collision translation shape sweep diagnostic",
        filter_path=filter_path,
        expected_max_hits=expected_max_hits,
        maximum_distance_m=expected_distance,
        require_available=exact_query_required,
    )
    exact_paired_hit_count = _validate_collision_sweep(
        evidence.get("exact_shape_sweep"),
        name="continuous collision exact shape sweep evidence",
        filter_path=filter_path,
        expected_max_hits=expected_max_hits,
        maximum_distance_m=expected_distance,
        require_available=exact_query_required,
    )
    if translation_shape_paired_hit_count != exact_paired_hit_count:
        raise HostedDroidError(
            "continuous collision translation and exact shape sweep evidence disagree"
        )
    reported_paired_hits = _contact_integer(
        evidence.get("paired_hit_count"),
        "continuous collision paired hit count",
        maximum=expected_max_hits,
    )
    if reported_paired_hits != paired_hit_count:
        raise HostedDroidError("continuous collision paired hit count is inconsistent")

    reported_exact_paired_hits = _contact_integer(
        evidence.get("exact_paired_hit_count"),
        "continuous collision exact paired hit count",
        maximum=expected_max_hits,
    )
    if reported_exact_paired_hits != exact_paired_hit_count:
        raise HostedDroidError(
            "continuous collision exact paired hit count is inconsistent"
        )
    broad_phase_only = paired_hit_count > 0 and exact_paired_hit_count == 0
    reported_broad_phase_only = evidence.get("broad_phase_only")
    if (
        type(reported_broad_phase_only) is not bool
        or reported_broad_phase_only is not broad_phase_only
    ):
        raise HostedDroidError(
            "continuous collision broad phase diagnostic is inconsistent"
        )

    swept_collision_risk_detected = (
        exact_paired_hit_count > 0
        and not previous_endpoint_contact
        and not current_endpoint_contact
    )
    reported_risk = evidence.get("swept_collision_risk_detected")
    tunneling_compatibility_alias = evidence.get("tunneling_detected")
    if (
        type(reported_risk) is not bool
        or type(tunneling_compatibility_alias) is not bool
        or reported_risk is not swept_collision_risk_detected
        or tunneling_compatibility_alias is not reported_risk
    ):
        raise HostedDroidError(
            "continuous collision swept collision risk verdict is inconsistent"
        )
    expected_classification = (
        "paired_tunneling"
        if swept_collision_risk_detected
        else "conservative_envelope_only"
        if broad_phase_only
        else "clear"
    )
    expected_passed = not swept_collision_risk_detected
    expected_failure_reasons = (
        ["paired_body_sweep_hit_without_endpoint_contact"]
        if swept_collision_risk_detected
        else []
    )
    failure_reasons = _contact_string_list(
        evidence.get("failure_reasons"),
        "continuous collision failure reasons",
        maximum=16,
    )
    evidence_errors = _contact_string_list(
        evidence.get("errors"),
        "continuous collision errors",
        maximum=16,
    )
    if evidence.get("complete") is not True or evidence_errors:
        raise HostedDroidError("continuous collision evidence is incomplete")
    if (
        evidence.get("classification") != expected_classification
        or evidence.get("passed") is not expected_passed
        or failure_reasons != expected_failure_reasons
    ):
        raise HostedDroidError("continuous collision classification is inconsistent")

    return _ContinuousCollisionEvidence(
        previous_sensor_pose=previous_sensor_pose,
        current_sensor_pose=current_sensor_pose,
        previous_filter_pose=previous_filter_pose,
        current_filter_pose=current_filter_pose,
        previous_endpoint_contact=previous_endpoint_contact,
        current_endpoint_contact=current_endpoint_contact,
        relative_translation_m=expected_distance,
        sensor_rotation_rad=sensor_rotation,
        filter_rotation_rad=filter_rotation,
        relative_rotation_rad=relative_rotation,
        rotation_envelope_inflation_m=inflation,
        swept_collision_risk_detected=swept_collision_risk_detected,
        paired_hit_count=paired_hit_count,
        exact_paired_hit_count=exact_paired_hit_count,
        broad_phase_only=broad_phase_only,
    )


class _DroidTaskSuccessTracker:
    """Own the temporal proof required for a policy-driven placement."""

    def __init__(
        self,
        spec: DroidTaskSuccessSpec,
        initial_state: _DroidTaskState,
    ) -> None:
        self.spec = spec
        self.initial_state = initial_state
        self.lift_observed = False
        self.consecutive_lift_checks = 0
        self.consecutive_settled_checks = 0
        self.checks = 0
        self.success_action_index: int | None = None
        self.previous_object_center = initial_state.object_bounds.center
        self.trajectory_valid = True
        self.hard_body_integrity_valid = True
        self.hard_body_integrity_reason: str | None = None
        self.release_command_after_lift_seen = False
        self.closed_support_observed = False
        self.minimum_closed_gripper_object_distance: float | None = None

    def initial_evaluation(self) -> dict[str, Any]:
        return {
            "predicate": self.spec.name,
            "phase": "initial",
            "policy_driven_lift_observed": False,
            "trajectory_valid": True,
            "hard_body_integrity": "pending",
            "hard_body_integrity_reason": None,
            "release_command_after_lift_seen": False,
            "closed_support_observed": False,
            "consecutive_lift_checks": 0,
            "required_lift_checks": self.spec.minimum_lift_checks,
            "consecutive_settled_checks": 0,
            "required_settled_checks": self.spec.required_settled_checks,
            "success": False,
        }

    def evaluate(
        self,
        state: _DroidTaskState,
        *,
        action_index: int,
        observed_gripper_position: float,
        commanded_gripper_closed: bool,
        contact_integrity: Mapping[str, Any],
        expected_contact_updates: int,
        expected_contact_physics_dt_seconds: float,
    ) -> dict[str, Any]:
        self.checks += 1
        contacts = self._contact_evidence(
            contact_integrity,
            expected_updates=expected_contact_updates,
            expected_physics_dt_seconds=expected_contact_physics_dt_seconds,
        )
        geometry = self._geometry(state)
        previous_object_center = self.previous_object_center
        object_displacement = math.dist(
            state.object_bounds.center,
            previous_object_center,
        )
        self.previous_object_center = state.object_bounds.center
        gripper_object_distance = math.dist(
            state.object_runtime_position,
            state.gripper_reference_position,
        )
        displacement_valid = (
            object_displacement
            <= self.spec.maximum_object_displacement_per_action_meters
        )
        self.trajectory_valid = self.trajectory_valid and displacement_valid
        if contacts["maximum_penetration_meters"] > (
            self.spec.maximum_contact_penetration_meters
        ):
            self._invalidate_hard_body_integrity("excessive_contact_penetration")
        if contacts["maximum_normal_impulse_ns"] > (
            self.spec.maximum_contact_normal_impulse_ns
        ):
            self._invalidate_hard_body_integrity("excessive_contact_impulse")
        self.trajectory_valid = self.trajectory_valid and self.hard_body_integrity_valid
        lifted_meters = (
            state.object_bounds.center[2] - self.initial_state.object_bounds.center[2]
        )
        gripper_closed = observed_gripper_position >= self.spec.gripper_closed_threshold
        if (
            commanded_gripper_closed
            and gripper_closed
            and contacts["bilateral_finger_contact_at_end"]
        ):
            self.closed_support_observed = True
            if self.minimum_closed_gripper_object_distance is None:
                self.minimum_closed_gripper_object_distance = gripper_object_distance
            else:
                self.minimum_closed_gripper_object_distance = min(
                    self.minimum_closed_gripper_object_distance,
                    gripper_object_distance,
                )
        closed_support_loss = (
            max(
                0.0,
                gripper_object_distance - self.minimum_closed_gripper_object_distance,
            )
            if self.minimum_closed_gripper_object_distance is not None
            else 0.0
        )
        lift_condition = (
            commanded_gripper_closed
            and gripper_closed
            and lifted_meters >= self.spec.minimum_lift_meters
            and not geometry["geometrically_in_receptacle"]
            and contacts["bilateral_finger_contact_at_end"]
            and contacts["bilateral_finger_contact_fraction"]
            >= self.spec.minimum_bilateral_contact_fraction
            and contacts["maximum_bilateral_contact_gap_updates"]
            <= self.spec.maximum_bilateral_contact_gap_updates
            and contacts["bilateral_normal_dot_at_end"]
            <= self.spec.maximum_bilateral_normal_dot
            and self.trajectory_valid
            and self.hard_body_integrity_valid
        )
        if lift_condition:
            self.consecutive_lift_checks += 1
        else:
            self.consecutive_lift_checks = 0
        self.lift_observed = (
            self.lift_observed
            or self.consecutive_lift_checks >= self.spec.minimum_lift_checks
        )
        if self.lift_observed and not commanded_gripper_closed:
            self.release_command_after_lift_seen = True

        if (
            self.closed_support_observed
            and commanded_gripper_closed
            and closed_support_loss > self.spec.maximum_closed_support_loss_meters
        ):
            self._invalidate_hard_body_integrity(
                "closed_gripper_support_loss_before_release"
            )
        if (
            (self.lift_observed or self.closed_support_observed)
            and commanded_gripper_closed
            and geometry["geometrically_in_receptacle"]
            and not contacts["bilateral_finger_contact_at_end"]
        ):
            self._invalidate_hard_body_integrity(
                "object_entered_receptacle_before_release"
            )
        self.trajectory_valid = self.trajectory_valid and self.hard_body_integrity_valid

        object_linear_speed = math.sqrt(
            sum(value * value for value in state.object_linear_velocity)
        )
        object_angular_speed = math.sqrt(
            sum(value * value for value in state.object_angular_velocity)
        )
        receptacle_linear_speed = math.sqrt(
            sum(value * value for value in state.receptacle_linear_velocity)
        )
        receptacle_angular_speed = math.sqrt(
            sum(value * value for value in state.receptacle_angular_velocity)
        )
        settled = (
            object_linear_speed <= self.spec.maximum_linear_speed_mps
            and object_angular_speed <= self.spec.maximum_angular_speed_rps
            and receptacle_linear_speed <= self.spec.maximum_linear_speed_mps
            and receptacle_angular_speed <= self.spec.maximum_angular_speed_rps
        )
        released = observed_gripper_position <= self.spec.gripper_released_threshold
        candidate = (
            self.lift_observed
            and self.trajectory_valid
            and self.hard_body_integrity_valid
            and self.release_command_after_lift_seen
            and geometry["geometrically_in_receptacle"]
            and contacts["receptacle_contact_at_end"]
            and released
            and settled
        )
        if candidate:
            self.consecutive_settled_checks += 1
        else:
            self.consecutive_settled_checks = 0
        success = self.consecutive_settled_checks >= self.spec.required_settled_checks
        if success and self.success_action_index is None:
            self.success_action_index = action_index

        return {
            "predicate": self.spec.name,
            "phase": "post_action",
            "action_index": action_index,
            "observed_gripper_position": observed_gripper_position,
            "commanded_gripper_closed": commanded_gripper_closed,
            "gripper_closed": gripper_closed,
            "gripper_released": released,
            "release_command_after_lift_seen": self.release_command_after_lift_seen,
            "object_displacement_meters": object_displacement,
            "gripper_object_distance_meters": gripper_object_distance,
            "closed_support_observed": self.closed_support_observed,
            "closed_support_loss_meters": closed_support_loss,
            "object_displacement_valid": displacement_valid,
            "trajectory_valid": self.trajectory_valid,
            "object_lift_meters": lifted_meters,
            "lift_condition_this_check": lift_condition,
            "consecutive_lift_checks": self.consecutive_lift_checks,
            "required_lift_checks": self.spec.minimum_lift_checks,
            "policy_driven_lift_observed": self.lift_observed,
            **contacts,
            "hard_body_integrity": (
                "pass" if self.hard_body_integrity_valid else "violated"
            ),
            "hard_body_integrity_reason": self.hard_body_integrity_reason,
            **geometry,
            "object_linear_speed_mps": object_linear_speed,
            "object_angular_speed_rps": object_angular_speed,
            "receptacle_linear_speed_mps": receptacle_linear_speed,
            "receptacle_angular_speed_rps": receptacle_angular_speed,
            "settled": settled,
            "candidate_success": candidate,
            "consecutive_settled_checks": self.consecutive_settled_checks,
            "required_settled_checks": self.spec.required_settled_checks,
            "success": success,
        }

    def _invalidate_hard_body_integrity(self, reason: str) -> None:
        if self.hard_body_integrity_valid:
            self.hard_body_integrity_valid = False
            self.hard_body_integrity_reason = reason

    def terminal_failure_reason(self) -> str | None:
        if not self.hard_body_integrity_valid:
            return (
                "hard_body_integrity_violation:"
                f"{self.hard_body_integrity_reason or 'unknown'}"
            )
        if not self.trajectory_valid:
            return "trajectory_violation:object_displacement"
        return None

    def _contact_evidence(
        self,
        trace: Mapping[str, Any],
        *,
        expected_updates: int,
        expected_physics_dt_seconds: float,
    ) -> dict[str, Any]:
        if not isinstance(trace, Mapping):
            raise HostedDroidError("contact integrity telemetry is missing")
        schema_version = trace.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _CONTACT_INTEGRITY_SCHEMA_VERSION
        ):
            raise HostedDroidError("contact integrity telemetry schema is unsupported")
        if trace.get("capture_source") != _CONTACT_CAPTURE_SOURCE:
            raise HostedDroidError("contact integrity capture source is invalid")
        if trace.get("sampling_semantics") != _CONTINUOUS_COLLISION_SAMPLING_SEMANTICS:
            raise HostedDroidError("contact integrity sampling semantics are invalid")
        expected_updates = _contact_integer(
            expected_updates,
            "expected contact integrity update count",
            minimum=1,
        )
        expected_physics_dt_seconds = _contact_number(
            expected_physics_dt_seconds,
            "expected contact integrity physics dt",
        )
        if expected_physics_dt_seconds <= 0:
            raise HostedDroidError(
                "expected contact integrity physics dt must be positive"
            )

        request = _contact_integrity_request(self.spec)
        expected_continuous = cast(
            Mapping[str, Any],
            request["continuous_collision"],
        )
        configured_continuous = _contact_mapping(
            trace.get("continuous_collision"),
            "contact integrity continuous collision configuration",
        )
        if set(configured_continuous) != set(expected_continuous):
            raise HostedDroidError(
                "contact integrity continuous collision configuration is incomplete"
            )
        for name in (
            "maximum_sensor_rotation_rad",
            "maximum_filter_rotation_rad",
        ):
            actual = _contact_number(
                configured_continuous.get(name),
                f"contact integrity continuous collision {name}",
                nonnegative=True,
            )
            expected = _contact_number(
                expected_continuous.get(name),
                f"configured contact integrity continuous collision {name}",
                nonnegative=True,
            )
            if not _contact_numbers_close(actual, expected):
                raise HostedDroidError(
                    f"contact integrity continuous collision {name} is invalid"
                )
        configured_max_hits = _contact_integer(
            configured_continuous.get("max_hits_per_pair"),
            "contact integrity continuous collision maximum hits",
            minimum=1,
            maximum=_CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR,
        )
        if configured_max_hits != expected_continuous["max_hits_per_pair"]:
            raise HostedDroidError(
                "contact integrity continuous collision maximum hits are invalid"
            )

        expected_limits = {
            "maximum_penetration_m": self.spec.maximum_contact_penetration_meters,
            "maximum_normal_impulse_ns": self.spec.maximum_contact_normal_impulse_ns,
            "maximum_sensor_rotation_rad_per_update": (
                _CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS
            ),
            "maximum_filter_rotation_rad_per_update": (
                _CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS
            ),
            "unreported_swept_collisions": 0.0,
        }
        limits = _contact_mapping(trace.get("limits"), "contact integrity limits")
        if set(limits) != set(expected_limits):
            raise HostedDroidError("contact integrity limits are incomplete")
        for name, expected in expected_limits.items():
            value = _contact_number(
                limits.get(name),
                f"contact integrity limit {name}",
                nonnegative=True,
            )
            if not _contact_numbers_close(value, expected):
                raise HostedDroidError(f"contact integrity limit {name} is invalid")

        trace_physics_dt = _contact_number(
            trace.get("physics_dt_seconds"),
            "contact integrity physics dt",
        )
        if trace_physics_dt <= 0 or not _contact_numbers_close(
            trace_physics_dt,
            expected_physics_dt_seconds,
        ):
            raise HostedDroidError("contact integrity physics dt is invalid")
        requested = _contact_integer(
            trace.get("requested_updates"),
            "contact integrity requested update count",
            minimum=1,
        )
        captured = _contact_integer(
            trace.get("captured_updates"),
            "contact integrity captured update count",
            minimum=1,
        )
        samples = trace.get("samples")
        if (
            not isinstance(samples, list)
            or requested != expected_updates
            or captured != expected_updates
            or len(samples) != expected_updates
        ):
            raise HostedDroidError("contact integrity update count is invalid")

        expected_pairs = {
            cast(str, pair["label"]): (
                cast(str, pair["sensor_path"]),
                cast(str, pair["filter_path"]),
            )
            for pair in cast(list[dict[str, Any]], request["pairs"])
        }
        pair_count = len(expected_pairs)
        contact_slots = expected_updates * pair_count * _CONTACT_MAX_CONTACTS_PER_PAIR
        sweep_slots = (
            expected_updates * pair_count * _CONTINUOUS_COLLISION_MAX_HITS_PER_PAIR * 2
        )
        if contact_slots + sweep_slots > _CONTACT_TRACE_SLOT_BUDGET:
            raise HostedDroidError("contact integrity response budget is unbounded")

        trace_complete = trace.get("complete")
        if type(trace_complete) is not bool:
            raise HostedDroidError("contact integrity completeness verdict is invalid")
        errors = _contact_string_list(
            trace.get("errors"),
            "contact integrity errors",
            maximum=expected_updates * pair_count,
        )
        saturated_pairs = _contact_string_list(
            trace.get("saturated_pairs"),
            "contact integrity saturated pairs",
            maximum=pair_count,
        )
        incomplete_pairs = _contact_string_list(
            trace.get("continuous_collision_incomplete_pairs"),
            "contact integrity continuous collision incomplete pairs",
            maximum=pair_count,
        )
        violations = trace.get("violations")
        if (
            not isinstance(violations, list)
            or len(violations) > expected_updates * pair_count * 3
        ):
            raise HostedDroidError("contact integrity violations are invalid")

        maximum_penetration = 0.0
        maximum_normal_impulse = 0.0
        maximum_relative_translation = 0.0
        maximum_sensor_rotation = 0.0
        maximum_filter_rotation = 0.0
        maximum_relative_rotation = 0.0
        maximum_rotation_envelope_inflation = 0.0
        updates_with_contact = 0
        unreported_swept_collisions = 0
        bilateral_contact_updates = 0
        receptacle_contact_updates = 0
        last_contacts: set[str] = set()
        last_bilateral_normal_dot: float | None = None
        bilateral_contact_flags: list[bool] = []
        previous_continuous: dict[str, _ContinuousCollisionEvidence] = {}
        expected_violations: dict[tuple[int, str, str], tuple[float, float]] = {}

        for expected_update_index, sample_value in enumerate(samples):
            sample = _contact_mapping(
                sample_value,
                "contact integrity sample",
            )
            sample_update_index = _contact_integer(
                sample.get("update_index"),
                "contact integrity update index",
                maximum=expected_updates - 1,
            )
            if sample_update_index != expected_update_index:
                raise HostedDroidError("contact integrity update index is invalid")
            sample_physics_dt = _contact_number(
                sample.get("physics_dt_seconds"),
                "contact integrity sample physics dt",
            )
            if sample_physics_dt <= 0 or not _contact_numbers_close(
                sample_physics_dt,
                expected_physics_dt_seconds,
            ):
                raise HostedDroidError("contact integrity sample physics dt is invalid")
            pairs = sample.get("pairs")
            if not isinstance(pairs, list) or len(pairs) != pair_count:
                raise HostedDroidError("contact integrity pair list is invalid")
            by_label: dict[str, Mapping[str, Any]] = {}
            for pair_value in pairs:
                pair = _contact_mapping(pair_value, "contact integrity pair")
                label = pair.get("label")
                if not isinstance(label, str) or label in by_label:
                    raise HostedDroidError(
                        "contact integrity pair label is invalid or duplicated"
                    )
                by_label[label] = pair
            if set(by_label) != set(expected_pairs):
                raise HostedDroidError("contact integrity pair labels are incomplete")

            contacts_this_update: set[str] = set()
            mean_normals: dict[str, tuple[float, float, float]] = {}
            shared_poses: dict[tuple[str, str], _ContactPose] = {}
            for label, (sensor_path, filter_path) in expected_pairs.items():
                pair = by_label[label]
                if (
                    pair.get("sensor_path") != sensor_path
                    or pair.get("filter_path") != filter_path
                ):
                    raise HostedDroidError("contact integrity pair paths are invalid")
                if type(pair.get("buffer_saturated")) is not bool:
                    raise HostedDroidError(
                        "contact integrity pair saturation verdict is invalid"
                    )
                if pair.get("buffer_saturated") is True:
                    raise HostedDroidError(
                        "contact integrity contact buffer is saturated"
                    )
                contacts_list = pair.get("contacts")
                if (
                    not isinstance(contacts_list, list)
                    or len(contacts_list) >= _CONTACT_MAX_CONTACTS_PER_PAIR
                ):
                    raise HostedDroidError(
                        "contact integrity contacts are invalid or unbounded"
                    )
                contact_count = _contact_integer(
                    pair.get("contact_count"),
                    "contact integrity contact count",
                    maximum=_CONTACT_MAX_CONTACTS_PER_PAIR,
                )
                if (
                    contact_count != len(contacts_list)
                    or contact_count >= _CONTACT_MAX_CONTACTS_PER_PAIR
                ):
                    raise HostedDroidError(
                        "contact integrity contact count is inconsistent"
                    )
                if contacts_list:
                    contacts_this_update.add(label)

                normals: list[tuple[float, float, float]] = []
                pair_maximum_penetration = 0.0
                pair_maximum_impulse = 0.0
                pair_total_impulse = 0.0
                for contact_value in contacts_list:
                    contact = _contact_mapping(
                        contact_value,
                        "contact integrity contact",
                    )
                    _contact_vector(
                        contact.get("point_m"),
                        "contact integrity contact point",
                        length=3,
                    )
                    normal = _contact_vector(
                        contact.get("normal_filter_to_sensor"),
                        "contact integrity contact normal",
                        length=3,
                    )
                    normal_length = math.sqrt(
                        sum(component * component for component in normal)
                    )
                    if normal_length <= 1e-9:
                        raise HostedDroidError("contact integrity normal is degenerate")
                    normals.append(
                        cast(
                            tuple[float, float, float],
                            tuple(component / normal_length for component in normal),
                        )
                    )
                    signed_separation = _contact_number(
                        contact.get("signed_separation_m"),
                        "contact integrity signed separation",
                    )
                    penetration = _contact_number(
                        contact.get("penetration_m"),
                        "contact integrity penetration",
                        nonnegative=True,
                    )
                    if not _contact_numbers_close(
                        penetration,
                        max(0.0, -signed_separation),
                    ):
                        raise HostedDroidError(
                            "contact integrity penetration is inconsistent"
                        )
                    impulse = _contact_number(
                        contact.get("normal_impulse_ns"),
                        "contact integrity normal impulse",
                        nonnegative=True,
                    )
                    force = _contact_number(
                        contact.get("normal_force_n"),
                        "contact integrity normal force",
                        nonnegative=True,
                    )
                    if not _contact_numbers_close(
                        force,
                        impulse / sample_physics_dt,
                    ):
                        raise HostedDroidError(
                            "contact integrity normal force is inconsistent"
                        )
                    pair_maximum_penetration = max(
                        pair_maximum_penetration,
                        penetration,
                    )
                    pair_maximum_impulse = max(pair_maximum_impulse, impulse)
                    pair_total_impulse += impulse

                reported_pair_penetration = _contact_number(
                    pair.get("maximum_penetration_m"),
                    "contact integrity pair maximum penetration",
                    nonnegative=True,
                )
                reported_pair_impulse = _contact_number(
                    pair.get("maximum_normal_impulse_ns"),
                    "contact integrity pair maximum normal impulse",
                    nonnegative=True,
                )
                reported_pair_total_impulse = _contact_number(
                    pair.get("total_normal_impulse_ns"),
                    "contact integrity pair total normal impulse",
                    nonnegative=True,
                )
                if (
                    not _contact_numbers_close(
                        reported_pair_penetration,
                        pair_maximum_penetration,
                    )
                    or not _contact_numbers_close(
                        reported_pair_impulse,
                        pair_maximum_impulse,
                    )
                    or not _contact_numbers_close(
                        reported_pair_total_impulse,
                        pair_total_impulse,
                    )
                ):
                    raise HostedDroidError(
                        "contact integrity pair contact summary is inconsistent"
                    )
                maximum_penetration = max(
                    maximum_penetration,
                    pair_maximum_penetration,
                )
                maximum_normal_impulse = max(
                    maximum_normal_impulse,
                    pair_maximum_impulse,
                )

                friction_contacts = pair.get("friction_contacts")
                if (
                    not isinstance(friction_contacts, list)
                    or len(friction_contacts) >= _CONTACT_MAX_CONTACTS_PER_PAIR
                ):
                    raise HostedDroidError(
                        "contact integrity friction data is invalid or unbounded"
                    )
                for friction_value in friction_contacts:
                    friction = _contact_mapping(
                        friction_value,
                        "contact integrity friction contact",
                    )
                    _contact_vector(
                        friction.get("point_m"),
                        "contact integrity friction point",
                        length=3,
                    )
                    vector = _contact_vector(
                        friction.get("impulse_vector_ns"),
                        "contact integrity friction impulse",
                        length=3,
                    )
                    magnitude = _contact_number(
                        friction.get("impulse_magnitude_ns"),
                        "contact integrity friction impulse magnitude",
                        nonnegative=True,
                    )
                    expected_magnitude = math.sqrt(
                        sum(component * component for component in vector)
                    )
                    if not _contact_numbers_close(magnitude, expected_magnitude):
                        raise HostedDroidError(
                            "contact integrity friction impulse is inconsistent"
                        )

                continuous = _validate_continuous_collision_evidence(
                    pair.get("continuous_collision"),
                    label=label,
                    sensor_path=sensor_path,
                    filter_path=filter_path,
                    manifold_contact=bool(contacts_list),
                    expected_config=expected_continuous,
                    previous=previous_continuous.get(label),
                    shared_poses=shared_poses,
                )
                previous_continuous[label] = continuous
                maximum_relative_translation = max(
                    maximum_relative_translation,
                    continuous.relative_translation_m,
                )
                maximum_sensor_rotation = max(
                    maximum_sensor_rotation,
                    continuous.sensor_rotation_rad,
                )
                maximum_filter_rotation = max(
                    maximum_filter_rotation,
                    continuous.filter_rotation_rad,
                )
                maximum_relative_rotation = max(
                    maximum_relative_rotation,
                    continuous.relative_rotation_rad,
                )
                maximum_rotation_envelope_inflation = max(
                    maximum_rotation_envelope_inflation,
                    continuous.rotation_envelope_inflation_m,
                )
                if continuous.swept_collision_risk_detected:
                    unreported_swept_collisions += 1
                    expected_violations[
                        (
                            expected_update_index,
                            label,
                            "unreported_swept_collision",
                        )
                    ] = (float(continuous.paired_hit_count), 0.0)
                if pair.get("complete") is not True:
                    raise HostedDroidError("contact integrity pair is incomplete")
                if (
                    pair_maximum_penetration
                    > self.spec.maximum_contact_penetration_meters
                ):
                    expected_violations[
                        (
                            expected_update_index,
                            label,
                            "maximum_penetration_m",
                        )
                    ] = (
                        pair_maximum_penetration,
                        self.spec.maximum_contact_penetration_meters,
                    )
                if pair_maximum_impulse > self.spec.maximum_contact_normal_impulse_ns:
                    expected_violations[
                        (
                            expected_update_index,
                            label,
                            "maximum_normal_impulse_ns",
                        )
                    ] = (
                        pair_maximum_impulse,
                        self.spec.maximum_contact_normal_impulse_ns,
                    )
                if normals:
                    mean = tuple(
                        sum(normal[axis] for normal in normals) / len(normals)
                        for axis in range(3)
                    )
                    mean_length = math.sqrt(
                        sum(component * component for component in mean)
                    )
                    if mean_length <= 1e-9:
                        raise HostedDroidError(
                            "contact integrity mean normal is degenerate"
                        )
                    mean_normals[label] = cast(
                        tuple[float, float, float],
                        tuple(component / mean_length for component in mean),
                    )

            updates_with_contact += int(bool(contacts_this_update))
            bilateral_contact = {
                _LEFT_FINGER_CONTACT,
                _RIGHT_FINGER_CONTACT,
            }.issubset(contacts_this_update)
            bilateral_contact_flags.append(bilateral_contact)
            bilateral_normal_dot_this_update: float | None = None
            if bilateral_contact:
                bilateral_contact_updates += 1
                left_normal = mean_normals[_LEFT_FINGER_CONTACT]
                right_normal = mean_normals[_RIGHT_FINGER_CONTACT]
                bilateral_normal_dot_this_update = sum(
                    left * right
                    for left, right in zip(left_normal, right_normal, strict=True)
                )
            last_bilateral_normal_dot = bilateral_normal_dot_this_update
            if _RECEPTACLE_CONTACT in contacts_this_update:
                receptacle_contact_updates += 1
            last_contacts = contacts_this_update

        actual_violations: dict[tuple[int, str, str], tuple[float, float]] = {}
        for violation_value in violations:
            violation = _contact_mapping(
                violation_value,
                "contact integrity violation",
            )
            if set(violation) != {
                "update_index",
                "pair_label",
                "metric",
                "observed",
                "limit",
            }:
                raise HostedDroidError("contact integrity violation fields are invalid")
            update_index = _contact_integer(
                violation.get("update_index"),
                "contact integrity violation update index",
                maximum=expected_updates - 1,
            )
            pair_label = violation.get("pair_label")
            metric = violation.get("metric")
            if pair_label not in expected_pairs or metric not in {
                "maximum_penetration_m",
                "maximum_normal_impulse_ns",
                "unreported_swept_collision",
            }:
                raise HostedDroidError("contact integrity violation is invalid")
            key = (update_index, cast(str, pair_label), cast(str, metric))
            if key in actual_violations:
                raise HostedDroidError("contact integrity violation is duplicated")
            actual_violations[key] = (
                _contact_number(
                    violation.get("observed"),
                    "contact integrity violation observed value",
                    nonnegative=True,
                ),
                _contact_number(
                    violation.get("limit"),
                    "contact integrity violation limit",
                    nonnegative=True,
                ),
            )
        if set(actual_violations) != set(expected_violations):
            raise HostedDroidError("contact integrity violations are inconsistent")
        for key, (expected_observed, expected_limit) in expected_violations.items():
            actual_observed, actual_limit = actual_violations[key]
            if not _contact_numbers_close(
                actual_observed, expected_observed
            ) or not _contact_numbers_close(actual_limit, expected_limit):
                raise HostedDroidError(
                    "contact integrity violation values are inconsistent"
                )

        summary = _contact_mapping(
            trace.get("summary"),
            "contact integrity summary",
        )
        expected_summary_fields = {
            "updates_with_contact",
            "maximum_penetration_m",
            "maximum_normal_impulse_ns",
            "unreported_swept_collisions",
            "maximum_relative_translation_m",
            "maximum_sensor_rotation_rad",
            "maximum_filter_rotation_rad",
            "maximum_relative_rotation_rad",
            "maximum_rotation_envelope_inflation_m",
        }
        if set(summary) != expected_summary_fields:
            raise HostedDroidError("contact integrity summary is incomplete")
        if (
            _contact_integer(
                summary.get("updates_with_contact"),
                "contact integrity updates with contact",
                maximum=expected_updates,
            )
            != updates_with_contact
        ):
            raise HostedDroidError("contact integrity contact summary is inconsistent")
        if (
            _contact_integer(
                summary.get("unreported_swept_collisions"),
                "contact integrity unreported swept collision count",
                maximum=expected_updates * pair_count,
            )
            != unreported_swept_collisions
        ):
            raise HostedDroidError(
                "contact integrity swept collision summary is inconsistent"
            )
        for name, expected in (
            ("maximum_penetration_m", maximum_penetration),
            ("maximum_normal_impulse_ns", maximum_normal_impulse),
            ("maximum_relative_translation_m", maximum_relative_translation),
            ("maximum_sensor_rotation_rad", maximum_sensor_rotation),
            ("maximum_filter_rotation_rad", maximum_filter_rotation),
            ("maximum_relative_rotation_rad", maximum_relative_rotation),
            (
                "maximum_rotation_envelope_inflation_m",
                maximum_rotation_envelope_inflation,
            ),
        ):
            actual = _contact_number(
                summary.get(name),
                f"contact integrity summary {name}",
                nonnegative=True,
            )
            if not _contact_numbers_close(actual, expected):
                raise HostedDroidError(
                    f"contact integrity summary {name} is inconsistent"
                )

        computed_within_limits = trace_complete is True and not expected_violations
        if trace.get("within_configured_limits") is not computed_within_limits:
            raise HostedDroidError("contact integrity limit verdict is inconsistent")
        if trace_complete is not True or errors or saturated_pairs or incomplete_pairs:
            raise HostedDroidError("contact integrity telemetry is incomplete")
        if unreported_swept_collisions > 0:
            raise HostedDroidError(
                "contact integrity reported an unreported swept collision"
            )

        maximum_bilateral_gap = 0
        current_gap = 0
        for has_bilateral_contact in bilateral_contact_flags:
            if has_bilateral_contact:
                current_gap = 0
            else:
                current_gap += 1
                maximum_bilateral_gap = max(maximum_bilateral_gap, current_gap)

        return {
            "contact_integrity_complete": True,
            "continuous_collision_complete": True,
            "unreported_swept_collisions": unreported_swept_collisions,
            "maximum_relative_translation_meters": maximum_relative_translation,
            "maximum_sensor_rotation_radians": maximum_sensor_rotation,
            "maximum_filter_rotation_radians": maximum_filter_rotation,
            "maximum_allowed_sensor_rotation_radians": (
                _CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS
            ),
            "maximum_allowed_filter_rotation_radians": (
                _CONTINUOUS_COLLISION_MAX_ROTATION_RADIANS
            ),
            "maximum_penetration_meters": maximum_penetration,
            "maximum_allowed_penetration_meters": (
                self.spec.maximum_contact_penetration_meters
            ),
            "maximum_normal_impulse_ns": maximum_normal_impulse,
            "maximum_allowed_normal_impulse_ns": (
                self.spec.maximum_contact_normal_impulse_ns
            ),
            "bilateral_finger_contact_updates": bilateral_contact_updates,
            "bilateral_finger_contact_fraction": (
                bilateral_contact_updates / len(samples)
            ),
            "maximum_bilateral_contact_gap_updates": maximum_bilateral_gap,
            "bilateral_finger_contact_at_end": {
                _LEFT_FINGER_CONTACT,
                _RIGHT_FINGER_CONTACT,
            }.issubset(last_contacts),
            "bilateral_normal_dot_at_end": last_bilateral_normal_dot,
            "receptacle_contact_updates": receptacle_contact_updates,
            "receptacle_contact_at_end": _RECEPTACLE_CONTACT in last_contacts,
        }

    def _geometry(self, state: _DroidTaskState) -> dict[str, Any]:
        object_bounds = state.object_bounds
        receptacle_bounds = state.receptacle_bounds
        object_center = object_bounds.center
        receptacle_center = receptacle_bounds.center
        object_size = object_bounds.size
        receptacle_size = receptacle_bounds.size
        raw_allowed_x = (
            receptacle_size[0] - object_size[0]
        ) / 2.0 - self.spec.horizontal_containment_margin_meters
        raw_allowed_y = (
            receptacle_size[1] - object_size[1]
        ) / 2.0 - self.spec.horizontal_containment_margin_meters
        allowed_x = raw_allowed_x * self.spec.horizontal_containment_fraction
        allowed_y = raw_allowed_y * self.spec.horizontal_containment_fraction
        offset_x = abs(object_center[0] - receptacle_center[0])
        offset_y = abs(object_center[1] - receptacle_center[1])
        if allowed_x <= 0 or allowed_y <= 0:
            normalized_radial_offset = None
            horizontally_contained = False
        else:
            normalized_radial_offset = math.sqrt(
                (offset_x / allowed_x) ** 2 + (offset_y / allowed_y) ** 2
            )
            horizontally_contained = normalized_radial_offset <= 1.0
        vertically_inserted = (
            object_bounds.minimum[2]
            >= receptacle_bounds.minimum[2] - self.spec.vertical_tolerance_meters
            and object_bounds.minimum[2]
            <= receptacle_bounds.maximum[2] - self.spec.minimum_insertion_meters
        )
        receptacle_size_change = max(
            abs(current - initial)
            for current, initial in zip(
                receptacle_size,
                self.initial_state.receptacle_bounds.size,
                strict=True,
            )
        )
        receptacle_geometry_stable = (
            receptacle_size_change <= self.spec.maximum_receptacle_size_change_meters
        )
        return {
            "horizontal_offset_meters": [offset_x, offset_y],
            "horizontal_clearance_meters": [allowed_x, allowed_y],
            "normalized_radial_offset": normalized_radial_offset,
            "horizontally_contained": horizontally_contained,
            "vertically_inserted": vertically_inserted,
            "receptacle_size_change_meters": receptacle_size_change,
            "receptacle_geometry_stable": receptacle_geometry_stable,
            "geometrically_in_receptacle": (
                horizontally_contained
                and vertically_inserted
                and receptacle_geometry_stable
            ),
        }


def _mediapy_module() -> Any | None:
    try:
        return importlib.import_module("mediapy")
    except ModuleNotFoundError as exc:
        if exc.name != "mediapy":
            raise
        return None


def _require_video_backend() -> None:
    if _mediapy_module() is not None:
        return
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise HostedDroidError(
        "video recording requires mediapy or both ffmpeg and ffprobe; "
        "install a video backend before launching the hosted rollout"
    )


def _run_video_command(
    command: list[str], operation: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no diagnostic output").strip()
        raise HostedDroidError(f"{operation} failed: {detail}") from exc


def _write_video_with_ffmpeg(
    temporary_path: Path,
    frame_paths: list[Path],
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise HostedDroidError("ffmpeg fallback requires both ffmpeg and ffprobe")

    indexes = [int(path.stem.removeprefix("action-")) for path in frame_paths]
    expected_indexes = list(range(indexes[0], indexes[0] + len(indexes)))
    if indexes != expected_indexes:
        raise HostedDroidError("video source frame indexes are not contiguous")

    _run_video_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            str(indexes[0]),
            "-i",
            str(frame_paths[0].parent / "action-%05d.png"),
            "-frames:v",
            str(len(frame_paths)),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ],
        "ffmpeg encode",
    )
    _run_video_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-f",
            "null",
            "-",
        ],
        "ffmpeg decode validation",
    )
    probe = _run_video_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
            "-of",
            "json",
            str(temporary_path),
        ],
        "ffprobe validation",
    )
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        shape = (int(stream["height"]), int(stream["width"]))
        frame_count = int(stream["nb_read_frames"])
        codec = stream["codec_name"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HostedDroidError("ffprobe returned incomplete video metadata") from exc
    if shape != (height, width) or frame_count != len(frame_paths) or codec != "h264":
        raise HostedDroidError(
            "encoded rollout video failed probe validation: "
            f"shape={shape}, frames={frame_count}, codec={codec}"
        )


def finalize_hosted_video_evidence(
    results_dir: Path,
    *,
    fps: int,
    source_camera: str,
) -> dict[str, Any] | None:
    """Encode and validate already-persisted rollout frames without rerunning Isaac."""

    video_frames_dir = results_dir / "video-frames"
    frame_paths = sorted(video_frames_dir.glob("action-*.png"))
    if not frame_paths:
        return None
    _require_video_backend()

    frames: list[np.ndarray] = []
    frame_manifest: list[dict[str, Any]] = []
    for path in frame_paths:
        raw = path.read_bytes()
        frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        frames.append(frame)
        frame_manifest.append(
            {
                "path": str(path.relative_to(results_dir)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    height, width = frames[0].shape[:2]
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise HostedDroidError("video source frames do not share one RGB shape")

    manifest_path = video_frames_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        (
            json.dumps(
                {
                    "schema_version": _EVIDENCE_SCHEMA_VERSION,
                    "source_camera": source_camera,
                    "frames": frame_manifest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    video_path = results_dir / "rollout.mp4"
    temporary_path = video_path.with_suffix(".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    mediapy = _mediapy_module()
    if mediapy is not None:
        mediapy.write_video(temporary_path, frames, fps=fps, codec="h264")
        decoded = np.asarray(mediapy.read_video(temporary_path))
        if decoded.shape != (len(frames), height, width, 3):
            raise HostedDroidError(
                "encoded rollout video failed decode validation: "
                f"expected {[len(frames), height, width, 3]}, got {list(decoded.shape)}"
            )
    else:
        _write_video_with_ffmpeg(
            temporary_path,
            frame_paths,
            fps=fps,
            width=width,
            height=height,
        )
    os.replace(temporary_path, video_path)
    video_bytes = video_path.read_bytes()
    return {
        "path": str(video_path.relative_to(results_dir)),
        "bytes": len(video_bytes),
        "sha256": hashlib.sha256(video_bytes).hexdigest(),
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps,
        "width": width,
        "height": height,
        "codec": "h264",
        "source_camera": source_camera,
        "source_frames_manifest": str(manifest_path.relative_to(results_dir)),
    }


def recover_hosted_video_evidence(results_dir: Path) -> dict[str, Any]:
    """Finalize video for a completed rollout whose local post-processing failed."""

    config_path = results_dir / "config.json"
    actions_path = results_dir / "actions.jsonl"
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = config_payload["config"]
        source_camera = config["cameras"][0]["prim_path"]
        fps = int(config["video_fps"])
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise HostedDroidError(
            "hosted evidence config is incomplete or invalid"
        ) from exc
    if fps < 1:
        raise HostedDroidError("hosted evidence video_fps must be positive")

    counts: dict[str, int] = {}
    try:
        for line in actions_path.read_text(encoding="utf-8").splitlines():
            record_type = json.loads(line).get("record_type")
            if not isinstance(record_type, str):
                raise HostedDroidError("action evidence record is missing record_type")
            counts[record_type] = counts.get(record_type, 0) + 1
    except (OSError, json.JSONDecodeError) as exc:
        raise HostedDroidError(
            "hosted action evidence is incomplete or invalid"
        ) from exc

    video = finalize_hosted_video_evidence(
        results_dir,
        fps=fps,
        source_camera=source_camera,
    )
    if video is None:
        raise HostedDroidError("hosted evidence does not contain rollout video frames")

    original_status = None
    error_path = results_dir / "error.json"
    if error_path.is_file():
        try:
            original_status = json.loads(error_path.read_text(encoding="utf-8")).get(
                "status"
            )
        except (OSError, json.JSONDecodeError):
            original_status = "unreadable"
    recovery = {
        "schema_version": _EVIDENCE_SCHEMA_VERSION,
        "status": "video_recovered",
        "recovered_at": _utc_now(),
        "original_status": original_status,
        "action_records": counts,
        "video": video,
    }
    _atomic_write(
        results_dir / "video-recovery.json",
        (json.dumps(recovery, indent=2, sort_keys=True) + "\n").encode(),
    )
    finalize_hosted_evidence_manifest(
        results_dir,
        terminal_record=("error.json" if error_path.is_file() else None),
    )
    return recovery


def _artifact_inventory(results_dir: Path) -> dict[str, dict[str, Any]]:
    root = results_dir.expanduser().resolve()
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HostedDroidError(
                f"hosted evidence must not contain symlinks: {path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == _EVIDENCE_MANIFEST_NAME:
            continue
        raw = path.read_bytes()
        files[relative] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return files


def _runtime_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    revision = os.environ.get("SIM_EVALS_REVISION")
    if revision is None:
        try:
            revision = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = None

    def package_version(name: str) -> str | None:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    provenance = {
        "sim_evals_version": package_version("sim-evals"),
        "cybernetics_sdk_version": package_version("cybernetic-physics"),
    }
    if revision:
        return {
            "status": "known",
            "sim_evals_revision": revision,
            **provenance,
        }
    return {
        "status": "unknown",
        "reason": "runtime did not expose a sim-evals revision",
        **provenance,
    }


def _unknown_artifact_provenance(reason: str) -> dict[str, str]:
    return {"status": "unknown", "reason": reason}


def _validate_manifest_provenance(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HostedDroidError(f"hosted evidence {name} provenance is invalid")
    status = value.get("status")
    if status == "known":
        revision = value.get("sim_evals_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise HostedDroidError(
                f"hosted evidence {name} provenance requires a revision"
            )
        for key in ("sim_evals_version", "cybernetics_sdk_version"):
            version = value.get(key)
            if version is not None and not isinstance(version, str):
                raise HostedDroidError(
                    f"hosted evidence {name} provenance has invalid {key}"
                )
    elif status == "unknown":
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise HostedDroidError(
                f"hosted evidence {name} provenance requires an unknown reason"
            )
    else:
        raise HostedDroidError(f"hosted evidence {name} provenance has invalid status")
    return cast(dict[str, Any], value)


def _manifest_identity(
    *,
    terminal_record: str,
    terminal_semantics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    files: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": _EVIDENCE_SCHEMA_VERSION,
        "terminal_record": terminal_record,
        "terminal_semantics": terminal_semantics,
        "provenance": provenance,
        "files": files,
    }


def _manifest_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _terminal_evidence_semantics(root: Path, terminal_record: str) -> dict[str, str]:
    try:
        payload = json.loads((root / terminal_record).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HostedDroidError("hosted evidence terminal record is invalid") from exc
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or type(schema_version) is not int
        or schema_version < 1
        or schema_version > _EVIDENCE_SCHEMA_VERSION
    ):
        raise HostedDroidError("hosted evidence terminal record has an invalid schema")
    if terminal_record == "result.json":
        result = payload.get("result")
        if payload.get("status") != "succeeded" or not isinstance(result, dict):
            raise HostedDroidError("hosted result terminal semantics are invalid")
        predicate = result.get("task_success_predicate")
        task_success = result.get("task_success")
        if predicate is None and task_success is None:
            expected_task_status = "not_evaluated"
        elif isinstance(predicate, str) and type(task_success) is bool:
            expected_task_status = "passed" if task_success else "failed"
        else:
            raise HostedDroidError("hosted result task semantics are invalid")
        execution_status = payload.get("execution_status", "completed")
        task_status = payload.get("task_status", expected_task_status)
        if execution_status != "completed" or task_status != expected_task_status:
            raise HostedDroidError("hosted result statuses are internally inconsistent")
    else:
        if payload.get("status") != "failed":
            raise HostedDroidError("hosted error terminal semantics are invalid")
        execution_status = payload.get("execution_status", "failed")
        task_status = payload.get("task_status", "not_evaluated")
        if execution_status != "failed" or task_status != "not_evaluated":
            raise HostedDroidError("hosted error statuses are internally inconsistent")
    return {
        "legacy_status": cast(str, payload["status"]),
        "execution_status": cast(str, execution_status),
        "task_status": cast(str, task_status),
    }


def finalize_hosted_evidence_manifest(
    results_dir: Path,
    *,
    terminal_record: str | None = None,
    artifact_producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the manifest last so every durable artifact is hash-bound."""

    root = results_dir.expanduser().resolve()
    present_terminal_records = [
        name for name in ("result.json", "error.json") if (root / name).is_file()
    ]
    if len(present_terminal_records) != 1:
        raise HostedDroidError(
            "hosted evidence requires exactly one result.json or error.json"
        )
    if terminal_record is None:
        terminal_record = present_terminal_records[0]
    elif terminal_record != present_terminal_records[0]:
        raise HostedDroidError(
            "hosted evidence terminal record does not match the durable record"
        )
    files = _artifact_inventory(root)
    provenance = {
        "artifact_producer": _validate_manifest_provenance(
            artifact_producer
            or _unknown_artifact_provenance(
                "historical finalization did not identify the artifact producer"
            ),
            "artifact producer",
        ),
        "manifest_writer": _validate_manifest_provenance(
            _runtime_provenance(),
            "manifest writer",
        ),
    }
    terminal_semantics = _terminal_evidence_semantics(root, terminal_record)
    identity = _manifest_identity(
        terminal_record=terminal_record,
        terminal_semantics=terminal_semantics,
        provenance=provenance,
        files=files,
    )
    manifest = {
        **identity,
        "created_at": _utc_now(),
        "aggregate_sha256": _manifest_identity_sha256(identity),
    }
    _atomic_write(
        root / _EVIDENCE_MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def verify_hosted_evidence_manifest(results_dir: Path) -> dict[str, Any]:
    """Verify the manifest inventory exactly, including unlisted-file rejection."""

    root = results_dir.expanduser().resolve()
    manifest_path = root / _EVIDENCE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HostedDroidError(
            "hosted evidence manifest is missing or invalid"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _EVIDENCE_SCHEMA_VERSION
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(manifest.get("aggregate_sha256"), str)
        or manifest.get("terminal_record") not in {"result.json", "error.json"}
        or not isinstance(manifest.get("terminal_semantics"), dict)
        or not isinstance(manifest.get("provenance"), dict)
    ):
        raise HostedDroidError("hosted evidence manifest has an invalid schema")
    provenance = cast(dict[str, Any], manifest["provenance"])
    _validate_manifest_provenance(
        provenance.get("artifact_producer"),
        "artifact producer",
    )
    _validate_manifest_provenance(
        provenance.get("manifest_writer"),
        "manifest writer",
    )
    files = _artifact_inventory(root)
    if manifest["files"] != files:
        raise HostedDroidError("hosted evidence manifest file inventory mismatch")
    terminal_record = cast(str, manifest["terminal_record"])
    present_terminal_records = {
        name for name in ("result.json", "error.json") if name in files
    }
    if present_terminal_records != {terminal_record}:
        raise HostedDroidError(
            "hosted evidence manifest requires exactly one terminal record"
        )
    if terminal_record not in files:
        raise HostedDroidError("hosted evidence manifest terminal record is missing")
    terminal_semantics = _terminal_evidence_semantics(root, terminal_record)
    if manifest["terminal_semantics"] != terminal_semantics:
        raise HostedDroidError("hosted evidence terminal semantics mismatch")
    identity = _manifest_identity(
        terminal_record=terminal_record,
        terminal_semantics=terminal_semantics,
        provenance=provenance,
        files=files,
    )
    aggregate = _manifest_identity_sha256(identity)
    if manifest["aggregate_sha256"] != aggregate:
        raise HostedDroidError("hosted evidence manifest identity mismatch")
    return manifest


class _EvidenceRecorder:
    def __init__(self, results_dir: Path, config: HostedDroidConfig) -> None:
        self.results_dir = results_dir
        self.config = config
        self.frames_dir = results_dir / "frames"
        self.samples_dir = results_dir / "samples"
        self.actions_path = results_dir / "actions.jsonl"
        self.task_states_path = results_dir / "task-states.jsonl"
        self.video_frames_dir = results_dir / "video-frames"
        self.video_path = results_dir / "rollout.mp4"
        self.video_manifest_path = self.video_frames_dir / "manifest.json"
        self.evidence_manifest_path = results_dir / _EVIDENCE_MANIFEST_NAME
        self.runtime_path = results_dir / "runtime.json"
        self._video_metadata: dict[str, Any] | None = None
        self._action_record_count = 0
        self._task_state_record_count = 0
        self.started_at = _utc_now()
        self._artifact_producer = _runtime_provenance()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.video_frames_dir.mkdir(parents=True, exist_ok=True)
        self._clear_previous_evidence()
        self._write_json(
            self.results_dir / "config.json",
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "created_at": self.started_at,
                "config": _config_dict(config),
            },
        )

    def write_frame(self, sample_index: int, camera_index: int, raw: bytes) -> None:
        name = _evidence_frame_name(sample_index, camera_index)
        _atomic_write(self.frames_dir / name, raw)

    def write_video_frame(self, action_index: int, rgb: np.ndarray) -> None:
        output = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            output, format="PNG"
        )
        _atomic_write(
            self.video_frames_dir / f"action-{action_index:05d}.png",
            output.getvalue(),
        )

    def write_runtime_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._write_json(
            self.runtime_path,
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                **metadata,
            },
        )

    def write_task_state(
        self,
        *,
        phase: str,
        action_index: int | None,
        state: _DroidTaskState,
        evaluation: Mapping[str, Any],
    ) -> None:
        record = {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "record_type": "task_state",
            "phase": phase,
            "action_index": action_index,
            "capture_method": (
                f"read_only_usd_bounds_and_{state.velocity_source}_rigid_state"
            ),
            "state": state.to_dict(),
            "evaluation": dict(evaluation),
        }
        self._append_jsonl(self.task_states_path, record)
        self._task_state_record_count += 1

    def finalize_video(self, fps: int) -> None:
        self._video_metadata = finalize_hosted_video_evidence(
            self.results_dir,
            fps=fps,
            source_camera=self.config.cameras[0].prim_path,
        )

    def write_sample(
        self,
        sample_index: int,
        response: Any,
        sampled_action_chunk: np.ndarray,
        action_chunk: np.ndarray,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "record_type": "sample",
            "sample_index": sample_index,
            "sampled_action_chunk_shape": list(sampled_action_chunk.shape),
            "sampled_action_chunk": sampled_action_chunk.astype(float).tolist(),
            "action_chunk": action_chunk.astype(float).tolist(),
        }
        predicted_video = _response_field(response, "predicted_video")
        if predicted_video is None:
            predicted_video = _response_field(response, "video")
        if predicted_video is not None:
            array = _tensor_array(predicted_video, "predicted_video")
            path = self.samples_dir / f"sample-{sample_index:05d}-predicted-video.npy"
            output = io.BytesIO()
            np.save(output, array, allow_pickle=False)
            _atomic_write(path, output.getvalue())
            record["predicted_video"] = {
                "path": str(path.relative_to(self.results_dir)),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        policy_metadata = _response_field(response, "policy_metadata")
        if policy_metadata is not None:
            if not isinstance(policy_metadata, Mapping):
                raise HostedDroidError("policy_metadata must be a mapping")
            record["policy_metadata"] = dict(policy_metadata)

        trajectory = _response_field(response, "trajectory")
        if trajectory is not None:
            if not isinstance(trajectory, list):
                raise HostedDroidError("trajectory must be a list of tensor mappings")
            arrays: dict[str, np.ndarray] = {}
            steps: list[dict[str, Any]] = []
            for step_index, step in enumerate(trajectory):
                if not isinstance(step, Mapping):
                    raise HostedDroidError(
                        f"trajectory step {step_index} must be a tensor mapping"
                    )
                step_metadata: dict[str, Any] = {}
                for key, value in step.items():
                    array = _tensor_array(value, f"trajectory[{step_index}].{key}")
                    source_key = str(key)
                    encoded_key = source_key.encode("utf-8").hex()
                    archive_key = f"step_{step_index:03d}__key_{encoded_key}"
                    arrays[archive_key] = array
                    step_metadata[source_key] = {
                        "archive_key": archive_key,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    }
                steps.append(step_metadata)
            path = self.samples_dir / f"sample-{sample_index:05d}-trajectory.npz"
            output = io.BytesIO()
            np.savez_compressed(
                output,
                **arrays,  # pyright: ignore[reportArgumentType]
            )
            _atomic_write(path, output.getvalue())
            record["trajectory"] = {
                "path": str(path.relative_to(self.results_dir)),
                "steps": steps,
            }
        self._append_action_record(record)

    def write_applied_action(
        self,
        *,
        sample_index: int,
        chunk_index: int,
        action_index: int,
        policy_action: np.ndarray,
        joint_positions: list[float],
        joint_indices: list[int],
        simulation_timing: Mapping[str, Any],
    ) -> None:
        self._append_action_record(
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "record_type": "applied_action",
                "sample_index": sample_index,
                "chunk_index": chunk_index,
                "action_index": action_index,
                "policy_action": policy_action.astype(float).tolist(),
                "joint_positions": joint_positions,
                "joint_indices": joint_indices,
                "simulation_timing": dict(simulation_timing),
            }
        )

    def write_action_target(
        self,
        joint_positions: list[float],
        joint_indices: list[int],
        *,
        sample_index: int,
        chunk_index: int,
        action_index: int,
        policy_action: np.ndarray,
    ) -> None:
        self._append_action_record(
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "record_type": "action_target",
                "sample_index": sample_index,
                "chunk_index": chunk_index,
                "action_index": action_index,
                "policy_action": policy_action.astype(float).tolist(),
                "joint_positions": joint_positions,
                "joint_indices": joint_indices,
            }
        )

    def _append_action_record(self, record: Mapping[str, Any]) -> None:
        self._append_jsonl(self.actions_path, record)
        self._action_record_count += 1

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("actions.jsonl write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_result(self, result: HostedDroidRunResult) -> None:
        task_status = (
            "not_evaluated"
            if result.task_success_predicate is None
            else "passed"
            if result.task_success
            else "failed"
        )
        self._write_json(
            self.results_dir / "result.json",
            {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "status": "succeeded",
                "execution_status": "completed",
                "task_status": task_status,
                "started_at": self.started_at,
                "finished_at": _utc_now(),
                "result": result.to_dict(),
                "evidence": self._evidence_dict(),
            },
        )
        finalize_hosted_evidence_manifest(
            self.results_dir,
            terminal_record="result.json",
            artifact_producer=self._artifact_producer,
        )

    def write_error(
        self,
        error: BaseException,
        session_id: str | None,
        *,
        evidence_errors: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "execution_status": "failed",
            "task_status": "not_evaluated",
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "evidence": self._evidence_dict(),
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if evidence_errors:
            payload["evidence_errors"] = evidence_errors
        self._write_json(self.results_dir / "error.json", payload)
        finalize_hosted_evidence_manifest(
            self.results_dir,
            terminal_record="error.json",
            artifact_producer=self._artifact_producer,
        )

    def _clear_previous_evidence(self) -> None:
        for path in (
            self.results_dir / "result.json",
            self.results_dir / "error.json",
            self.actions_path,
            self.task_states_path,
            self.runtime_path,
            self.evidence_manifest_path,
        ):
            path.unlink(missing_ok=True)
        for path in self.frames_dir.glob("sample-*.png"):
            path.unlink()
        for path in self.samples_dir.glob("sample-*.*"):
            path.unlink()
        for path in self.video_frames_dir.glob("action-*.png"):
            path.unlink()
        self.video_path.unlink(missing_ok=True)
        self.video_path.with_suffix(".tmp.mp4").unlink(missing_ok=True)
        self.video_manifest_path.unlink(missing_ok=True)

    def _evidence_dict(self) -> dict[str, Any]:
        frames = []
        for path in sorted(self.frames_dir.glob("sample-*.png")):
            stem = path.stem
            sample_index = int(stem.split("-", 2)[1])
            camera_name = stem.split("-", 2)[2]
            frames.append(
                {
                    "sample_index": sample_index,
                    "camera": camera_name,
                    "path": str(path.relative_to(self.results_dir)),
                }
            )
        sample_artifacts = [
            str(path.relative_to(self.results_dir))
            for path in sorted(self.samples_dir.glob("sample-*.*"))
        ]
        actions = None
        if self.actions_path.is_file():
            actions = {
                "path": str(self.actions_path.relative_to(self.results_dir)),
                "records": self._action_record_count,
            }
        task_states = None
        if self.task_states_path.is_file():
            task_states = {
                "path": str(self.task_states_path.relative_to(self.results_dir)),
                "records": self._task_state_record_count,
            }
        return {
            "artifact_manifest": _EVIDENCE_MANIFEST_NAME,
            "frames": frames,
            "actions": actions,
            "task_states": task_states,
            "sample_artifacts": sample_artifacts,
            "runtime": (
                str(self.runtime_path.relative_to(self.results_dir))
                if self.runtime_path.is_file()
                else None
            ),
            "video": self._video_metadata if self.video_path.is_file() else None,
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        _atomic_write(path, encoded)


ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_CLOSED_RADIANS = math.pi / 4
_DROID_INITIAL_ARM_JOINT_POSITIONS = (
    0.0,
    -math.pi / 5,
    0.0,
    -4 * math.pi / 5,
    0.0,
    3 * math.pi / 5,
    0.0,
)
_DROID_DYNAMICS_PROFILE = "cybernetics_droid_contact_v1"
_DROID_DYNAMICS_STDOUT_PREFIX = "DROID_DYNAMICS_PROFILE="
_DROID_GRIPPER_VELOCITY_LIMIT_RADIANS = 1.0
_DROID_GRIPPER_STIFFNESS = 100.0
_DROID_GRIPPER_DAMPING = 0.0002
_DROID_GRIPPER_MAX_FORCE = 16.5
_DROID_FINGER_STATIC_FRICTION = 1.5
_DROID_FINGER_DYNAMIC_FRICTION = 1.2
_DROID_CUBE_STATIC_FRICTION = 0.8
_DROID_CUBE_DYNAMIC_FRICTION = 0.6
_DROID_RECEPTACLE_STATIC_FRICTION = 0.6
_DROID_RECEPTACLE_DYNAMIC_FRICTION = 0.5
_DROID_TABLE_STATIC_FRICTION = 0.5
_DROID_TABLE_DYNAMIC_FRICTION = 0.4
_DROID_FRICTION_COMBINE_MODE = "average"
_DROID_CUBE_MASS_KG = 0.04
_DROID_CONTACT_OFFSET_METERS = 0.002
_DROID_REST_OFFSET_METERS = 0.0
_DROID_MAX_DEPENETRATION_VELOCITY_MPS = 3.0
_CAMERA_CAPTURE_ATTEMPTS = 10
_CAMERA_CAPTURE_RETRY_SECONDS = 0.5
_CAMERA_WARMUP_ACTION_PERIODS = 4
_CAMERA_WARMUP_SECONDS = 1.0
_INITIAL_SETTLE_MAX_CHECKS = 30
_INITIAL_SETTLE_STABLE_CHECKS = 2
_INITIAL_ARM_ERROR_TOLERANCE_RADIANS = 0.05
_INITIAL_ARM_MOTION_TOLERANCE_RADIANS = 0.005
_INITIAL_GRIPPER_TOLERANCE = 0.1
_REPLAY_INITIAL_ARM_TOLERANCE_RADIANS = 0.005
_REPLAY_INITIAL_GRIPPER_TOLERANCE = 0.01
_REPLAY_INITIAL_TASK_GEOMETRY_TOLERANCE_METERS = 0.001
_REPLAY_INITIAL_TASK_VELOCITY_TOLERANCE = 0.01
_MIN_LUMINANCE_P99 = 12.0
_MIN_LUMINANCE_STDDEV = 2.0
_MIN_NON_DARK_FRACTION = 0.01
_MIN_NON_WHITE_FRACTION = 0.02
_NON_WHITE_LUMINANCE_CUTOFF = 245.0

_REPLAY_TASK_GEOMETRY_PATHS = (
    ("object_bounds", "center"),
    ("object_bounds", "minimum"),
    ("object_bounds", "maximum"),
    ("object_bounds", "size"),
    ("receptacle_bounds", "center"),
    ("receptacle_bounds", "minimum"),
    ("receptacle_bounds", "maximum"),
    ("receptacle_bounds", "size"),
    ("object_runtime_position",),
    ("gripper_reference_position",),
)
_REPLAY_TASK_VELOCITY_PATHS = (
    ("object_linear_velocity",),
    ("object_angular_velocity",),
    ("receptacle_linear_velocity",),
    ("receptacle_angular_velocity",),
)


def _replay_task_state_vectors(
    value: object,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    if not isinstance(value, Mapping):
        raise ValueError("recorded replay initial task state must be an object")

    def vector(path: tuple[str, ...]) -> np.ndarray:
        current: object = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(
                    "recorded replay initial task state is missing " + ".".join(path)
                )
            current = current[key]
        try:
            array = np.asarray(current, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "recorded replay initial task state has invalid " + ".".join(path)
            ) from exc
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(
                "recorded replay initial task state has invalid " + ".".join(path)
            )
        return array

    geometry = {".".join(path): vector(path) for path in _REPLAY_TASK_GEOMETRY_PATHS}
    velocity = {".".join(path): vector(path) for path in _REPLAY_TASK_VELOCITY_PATHS}
    velocity_source = value.get("velocity_source")
    if not isinstance(velocity_source, str) or not velocity_source:
        raise ValueError("recorded replay initial task state lacks velocity provenance")
    return geometry, velocity, velocity_source


def _compare_replay_task_states(
    source: object,
    current: object,
) -> dict[str, Any]:
    source_geometry, source_velocity, source_velocity_kind = _replay_task_state_vectors(
        source
    )
    current_geometry, current_velocity, current_velocity_kind = (
        _replay_task_state_vectors(current)
    )
    if source_velocity_kind != current_velocity_kind:
        raise ValueError("recorded replay initial task velocity provenance changed")
    geometry_delta = max(
        float(np.max(np.abs(current_geometry[key] - source_geometry[key])))
        for key in source_geometry
    )
    velocity_delta = max(
        float(np.max(np.abs(current_velocity[key] - source_velocity[key])))
        for key in source_velocity
    )
    if geometry_delta > _REPLAY_INITIAL_TASK_GEOMETRY_TOLERANCE_METERS:
        raise ValueError(
            "recorded replay initial task geometry differs from its source: "
            f"maximum_delta={geometry_delta:.9f} m"
        )
    if velocity_delta > _REPLAY_INITIAL_TASK_VELOCITY_TOLERANCE:
        raise ValueError(
            "recorded replay initial task velocity differs from its source: "
            f"maximum_delta={velocity_delta:.9f}"
        )
    return {
        "source_velocity_provenance": source_velocity_kind,
        "maximum_geometry_delta_meters": geometry_delta,
        "maximum_velocity_delta": velocity_delta,
        "maximum_allowed_geometry_delta_meters": (
            _REPLAY_INITIAL_TASK_GEOMETRY_TOLERANCE_METERS
        ),
        "maximum_allowed_velocity_delta": _REPLAY_INITIAL_TASK_VELOCITY_TOLERANCE,
    }


def _droid_joint_positions_for_policy_action(action: np.ndarray) -> list[float]:
    """Map one validated DROID policy action to runtime joint targets."""

    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("DROID policy action must contain eight finite values")
    gripper = GRIPPER_CLOSED_RADIANS if float(action[7]) > 0.5 else 0.0
    return [*action[:7].astype(float).tolist(), gripper]


def _default_cameras() -> tuple[CameraSpec, ...]:
    generation = uuid.uuid4().hex[:12]
    external_root = f"/World/droid_eval_{generation}"
    wrist_parent = "/World/robot/Gripper/Robotiq_2F_85/base_link"
    return (
        CameraSpec(
            "exterior_1",
            f"{external_root}/external_cam",
            (0.05, 0.57, 0.66),
            (-0.393, -0.195, 0.399, 0.805),
            2.1,
        ),
        CameraSpec(
            "exterior_2",
            f"{external_root}/external_cam_2",
            (0.05, -0.57, 0.66),
            (0.805, 0.399, -0.195, -0.393),
            2.1,
        ),
        CameraSpec(
            "wrist",
            f"{wrist_parent}/droid_eval_wrist_cam_{generation}",
            (0.011, -0.031, -0.074),
            (-0.420, 0.570, 0.576, -0.409),
            2.8,
        ),
    )


def _viewer_camera_for(policy_camera: CameraSpec) -> CameraSpec:
    """Return a viewer-only clone that cannot mutate a policy observation."""

    return CameraSpec(
        "viewer",
        f"{policy_camera.prim_path.rsplit('/', 1)[0]}/{_DROID_VIEWER_CAMERA_NAME}",
        policy_camera.position,
        policy_camera.orientation_wxyz,
        policy_camera.focal_length,
        clipping_range=policy_camera.clipping_range,
        focus_distance=policy_camera.focus_distance,
        horizontal_aperture=policy_camera.horizontal_aperture,
        vertical_aperture=policy_camera.vertical_aperture,
    )


class HostedDroidRunner:
    """Run a Cybernetics DROID policy against a hosted Isaac MCP session."""

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
        self._validate_action_source()
        if self.config.record_video and self.config.results_dir is not None:
            _require_video_backend()
        evidence = (
            _EvidenceRecorder(self.config.results_dir, self.config)
            if self.config.results_dir is not None
            else None
        )
        session_id = self.config.session_id
        owns_session = False
        cleanup_errors: list[str] = []
        try:
            try:
                if session_id is None:
                    launch_kwargs: dict[str, Any] = {
                        "wait": False,
                    }
                    if self.config.runtime_provider is not None:
                        launch_kwargs["runtime_provider"] = self.config.runtime_provider
                    launch = self.simulation_client.launch(
                        self.config.environment_uri,
                        **launch_kwargs,
                    )
                    session_id = launch.session_id
                    owns_session = True
                    self.simulation_client.wait_for_session(
                        session_id,
                        timeout_seconds=self.config.launch_timeout_seconds,
                        poll_interval_seconds=self.config.readiness_poll_seconds,
                    )
                else:
                    self.simulation_client.wait_for_session(
                        session_id,
                        timeout_seconds=self.config.launch_timeout_seconds,
                        poll_interval_seconds=self.config.readiness_poll_seconds,
                    )
                mcp_session = getattr(self.simulation_client, "mcp_session", None)
                if not callable(mcp_session):
                    raise HostedDroidError(
                        "Cybernetics SimulationClient must expose "
                        "mcp_session(session_id, *, ttl_seconds)"
                    )
                mcp_session = cast(
                    Callable[..., AbstractContextManager[MCPClient]], mcp_session
                )
                with mcp_session(
                    session_id,
                    ttl_seconds=_HOSTED_MCP_CREDENTIAL_TTL_SECONDS,
                ) as mcp:
                    self._wait_for_isaac(mcp)
                    repaired_robot = self._ensure_robot(mcp)
                    runtime_dynamics = self._configure_robot_dynamics(mcp)
                    self._retire_previous_cameras(mcp)
                    created_cameras = self._ensure_cameras(mcp)
                    viewer_camera = self._ensure_viewer_camera(mcp)
                    self._set_viewer_camera(mcp, viewer_camera.prim_path)
                    joint_indices = self._joint_indices(mcp)
                    initial_control_source = self._reset_robot_for_policy(
                        mcp,
                        joint_indices,
                    )
                    physics_dt, physics_steps_per_action = self._control_cadence(mcp)
                    control_hz = 1.0 / (physics_dt * physics_steps_per_action)
                    initial_arm_positions, initial_gripper_position = (
                        self._settle_robot_for_policy(
                            mcp,
                            joint_indices,
                            physics_steps_per_action,
                        )
                    )
                    replay_initial_state = self._validate_replay_initial_state(
                        initial_arm_positions,
                        initial_gripper_position,
                    )
                    if evidence is not None:
                        evidence.write_runtime_metadata(
                            {
                                "base_model": self.config.base_model,
                                "physics_dt": physics_dt,
                                "physics_steps_per_action": physics_steps_per_action,
                                "physics_hz": self.config.physics_hz,
                                "solver_position_iterations": (
                                    self.config.solver_position_iterations
                                ),
                                "solver_velocity_iterations": (
                                    self.config.solver_velocity_iterations
                                ),
                                "target_control_hz": self.config.target_control_hz,
                                "control_hz": control_hz,
                                "action_source": self.config.action_source,
                                "replay_source_sha256": (
                                    self.config.replay_source_sha256
                                ),
                                "replay_actions_sha256": getattr(
                                    self.sampling_api,
                                    "actions_sha256",
                                    None,
                                ),
                                "replay_source_file_count": len(
                                    getattr(
                                        self.sampling_api,
                                        "source_files_sha256",
                                        {},
                                    )
                                ),
                                "replay_source_artifact_producer": getattr(
                                    self.sampling_api,
                                    "source_artifact_producer",
                                    None,
                                ),
                                "replay_initial_state": replay_initial_state,
                                "robot_dynamics_profile": _DROID_DYNAMICS_PROFILE,
                                "runtime_dynamics": runtime_dynamics,
                                "policy_camera_paths": list(created_cameras),
                                "policy_camera_roles": [
                                    camera.role for camera in self.config.cameras
                                ],
                                "policy_camera_calibration": (
                                    "validated_before_and_after_every_capture_bundle"
                                ),
                                "viewer_camera_path": viewer_camera.prim_path,
                                "viewer_camera_isolated_from_policy": True,
                                "initial_arm_joint_targets": list(
                                    _DROID_INITIAL_ARM_JOINT_POSITIONS
                                ),
                                "initial_arm_joint_positions": (
                                    initial_arm_positions.tolist()
                                ),
                                "initial_gripper_target": 0.0,
                                "initial_gripper_position": initial_gripper_position,
                                "joint_target_control_source": (initial_control_source),
                                "joint_measurement_source": "runtime_articulation",
                                "task_success_predicate": (
                                    self.config.task_success.name
                                    if self.config.task_success is not None
                                    else None
                                ),
                            }
                        )
                    self.sampling_api.reset_sampling_session()
                    rollout = self._rollout(
                        mcp,
                        joint_indices,
                        physics_steps_per_action,
                        evidence,
                    )
                result = HostedDroidRunResult(
                    session_id=session_id,
                    samples=rollout.samples,
                    action_steps=rollout.action_steps,
                    repaired_robot=repaired_robot,
                    created_cameras=tuple(created_cameras),
                    session_retained=not owns_session or self.config.keep_session,
                    physics_dt=physics_dt,
                    physics_steps_per_action=physics_steps_per_action,
                    control_hz=control_hz,
                    task_success=rollout.task_success,
                    task_success_predicate=rollout.task_success_predicate,
                    task_success_action_index=rollout.task_success_action_index,
                    task_success_checks=rollout.task_success_checks,
                    task_success_reason=rollout.task_success_reason,
                )
            finally:
                cleanup_errors.extend(self._cleanup(session_id, owns_session))
            if cleanup_errors:
                raise HostedDroidError(
                    "hosted DROID cleanup failed: " + "; ".join(cleanup_errors)
                )
            if evidence is not None and self.config.record_video:
                evidence.finalize_video(self.config.video_fps)
        except BaseException as exc:
            if evidence is not None:
                evidence_errors = list(cleanup_errors)
                if self.config.record_video and not evidence.video_path.is_file():
                    try:
                        evidence.finalize_video(self.config.video_fps)
                    except Exception as evidence_exc:
                        evidence_errors.append(
                            f"video finalization failed: {type(evidence_exc).__name__}: "
                            f"{evidence_exc}"
                        )
                evidence.write_error(
                    exc,
                    session_id,
                    evidence_errors=evidence_errors,
                )
            raise
        if evidence is not None:
            evidence.write_result(result)
        return result

    def _validate_action_source(self) -> None:
        if self.config.action_source != "recorded_replay":
            return
        if getattr(self.sampling_api, "action_source", None) != "recorded_replay":
            raise HostedDroidError(
                "recorded_replay requires an explicitly marked replay sampler"
            )
        if (
            getattr(self.sampling_api, "source_sha256", None)
            != self.config.replay_source_sha256
        ):
            raise HostedDroidError(
                "recorded replay sampler digest does not match its configuration"
            )

    def _validate_replay_initial_state(
        self,
        arm_positions: np.ndarray,
        gripper_position: float,
    ) -> dict[str, Any] | None:
        if self.config.action_source != "recorded_replay":
            return None
        try:
            expected_arm = np.asarray(
                getattr(
                    self.sampling_api,
                    "source_initial_arm_joint_positions",
                    None,
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            expected_arm = np.asarray([], dtype=np.float64)
        expected_gripper = getattr(
            self.sampling_api,
            "source_initial_gripper_position",
            None,
        )
        if (
            expected_arm.shape != (7,)
            or not np.isfinite(expected_arm).all()
            or isinstance(expected_gripper, bool)
            or not isinstance(expected_gripper, (int, float))
            or not math.isfinite(float(expected_gripper))
        ):
            raise HostedDroidError(
                "recorded replay source is missing finite initial robot state"
            )
        arm_delta = np.abs(np.asarray(arm_positions, dtype=np.float64) - expected_arm)
        gripper_delta = abs(float(gripper_position) - float(expected_gripper))
        maximum_arm_delta = float(np.max(arm_delta))
        if maximum_arm_delta > _REPLAY_INITIAL_ARM_TOLERANCE_RADIANS:
            raise HostedDroidError(
                "recorded replay initial arm state differs from its source: "
                f"maximum_delta={maximum_arm_delta:.9f} rad"
            )
        if gripper_delta > _REPLAY_INITIAL_GRIPPER_TOLERANCE:
            raise HostedDroidError(
                "recorded replay initial gripper state differs from its source: "
                f"delta={gripper_delta:.9f}"
            )
        return {
            "source_arm_joint_positions": expected_arm.tolist(),
            "source_gripper_position": float(expected_gripper),
            "maximum_arm_delta_radians": maximum_arm_delta,
            "gripper_delta": gripper_delta,
            "maximum_allowed_arm_delta_radians": (
                _REPLAY_INITIAL_ARM_TOLERANCE_RADIANS
            ),
            "maximum_allowed_gripper_delta": _REPLAY_INITIAL_GRIPPER_TOLERANCE,
        }

    def _validate_replay_initial_task_state(
        self,
        state: _DroidTaskState,
    ) -> dict[str, Any] | None:
        if self.config.action_source != "recorded_replay":
            return None
        source = getattr(self.sampling_api, "source_initial_task_state", None)
        try:
            return _compare_replay_task_states(source, state.to_dict())
        except ValueError as exc:
            raise HostedDroidError(str(exc)) from exc

    def _cleanup(
        self,
        session_id: str | None,
        owns_session: bool,
    ) -> list[str]:
        cleanup_errors: list[str] = []
        try:
            try:
                self.sampling_api.close()
            except Exception as exc:
                cleanup_errors.append(
                    f"sampling API close failed: {type(exc).__name__}: {exc}"
                )
        finally:
            if owns_session and session_id is not None and not self.config.keep_session:
                try:
                    self.simulation_client.stop_session(session_id)
                except Exception as exc:
                    cleanup_errors.append(
                        f"session stop failed: {type(exc).__name__}: {exc}"
                    )
        # A normal stop failure is evidence, but BaseException from sampler close
        # still propagates after the independent stop attempt above.
        return cleanup_errors

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
        metadata = self._robot_metadata(mcp)
        if _has_droid_joints(metadata):
            return False

        if metadata["prim_exists"]:
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
        repaired = self._robot_metadata(mcp)
        if not _has_droid_joints(repaired):
            raise HostedDroidError(
                f"loaded robot at {self.config.robot_prim_path} is not DROID-compatible"
            )
        return True

    def _robot_metadata(self, mcp: MCPClient) -> dict[str, Any]:
        robot_path = json.dumps(self.config.robot_prim_path)
        prefix = json.dumps(_ROBOT_METADATA_STDOUT_PREFIX)
        code = f"""
import json
import omni.usd
from pxr import Usd, UsdPhysics

stage = omni.usd.get_context().get_stage()
robot_path = {robot_path}
root = stage.GetPrimAtPath(robot_path)
joint_names = []
if root.IsValid():
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(
            UsdPhysics.PrismaticJoint
        ):
            joint_names.append(prim.GetName())
payload = {{
    "prim_exists": bool(root.IsValid()),
    "joint_names": joint_names,
    "num_dof": len(joint_names),
    "source": "usd_metadata",
}}
print({prefix} + json.dumps(payload, sort_keys=True))
"""
        response = self._call(mcp, "isaac.execute_script", {"code": code})
        stdout = response.get("stdout")
        if not isinstance(stdout, str):
            raise HostedDroidError(
                "isaac.execute_script did not return robot-metadata stdout"
            )
        encoded_metadata = next(
            (
                line.removeprefix(_ROBOT_METADATA_STDOUT_PREFIX)
                for line in reversed(stdout.splitlines())
                if line.startswith(_ROBOT_METADATA_STDOUT_PREFIX)
            ),
            None,
        )
        if encoded_metadata is None:
            raise HostedDroidError(
                "isaac.execute_script did not emit the robot-metadata marker"
            )
        try:
            payload = json.loads(encoded_metadata)
        except json.JSONDecodeError as exc:
            raise HostedDroidError("Isaac emitted invalid robot-metadata JSON") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("prim_exists"), bool
        ):
            raise HostedDroidError("Isaac emitted invalid robot metadata")
        joint_names = payload.get("joint_names")
        if not isinstance(joint_names, list) or not all(
            isinstance(name, str) for name in joint_names
        ):
            raise HostedDroidError("Isaac emitted invalid robot joint metadata")
        return cast(dict[str, Any], payload)

    def _configure_robot_dynamics(self, mcp: MCPClient) -> dict[str, Any]:
        physics_hz = self.config.physics_hz
        solver_position_iterations = self.config.solver_position_iterations
        solver_velocity_iterations = self.config.solver_velocity_iterations
        robot_path = json.dumps(self.config.robot_prim_path)
        cube_path = json.dumps("/World/rubiks_cube")
        receptacle_path = json.dumps(
            self.config.task_success.receptacle_prim_path
            if self.config.task_success is not None
            else "/World/_24_bowl"
        )
        table_path = json.dumps("/World/table")
        joint_settings = json.dumps(
            {
                **{
                    f"panda_joint{index}": {
                        "effort_limit": 87.0,
                        "velocity_limit_degrees": math.degrees(2.175),
                    }
                    for index in range(1, 5)
                },
                **{
                    f"panda_joint{index}": {
                        "effort_limit": 12.0,
                        "velocity_limit_degrees": math.degrees(2.61),
                    }
                    for index in range(5, 8)
                },
            },
            sort_keys=True,
        )
        gripper_joint_name = json.dumps(GRIPPER_JOINT_NAME)
        gripper_velocity_limit_degrees = math.degrees(
            _DROID_GRIPPER_VELOCITY_LIMIT_RADIANS
        )
        code = f"""
import carb
import json
import math
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.api import PhysicsContext
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
timeline.pause()
timeline.commit()
settings = carb.settings.get_settings()
settings.set("/app/player/useFixedTimeStepping", True)
settings.set("/app/player/CompensatePlayDelayInSecs", 0.0)
settings.set("/persistent/simulation/minFrameRate", int({physics_hz}))
timeline.set_play_every_frame(True)
timeline.set_ticks_per_frame(1)
timeline.set_time_codes_per_second({physics_hz})
robot_path = {robot_path}
cube_path = {cube_path}
receptacle_path = {receptacle_path}
table_path = {table_path}
finger_binding_paths = [
    robot_path + "/Gripper/Robotiq_2F_85/left_inner_finger",
    robot_path + "/Gripper/Robotiq_2F_85/right_inner_finger",
]
root = stage.GetPrimAtPath(robot_path)
if not root.IsValid():
    raise RuntimeError(f"DROID robot prim not found: {{robot_path}}")

def define_physics_material(path, static_friction, dynamic_friction):
    material = UsdShade.Material.Define(stage, path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(static_friction)
    material_api.CreateDynamicFrictionAttr(dynamic_friction)
    material_api.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(
        material.GetPrim()
    ).CreateFrictionCombineModeAttr({_DROID_FRICTION_COMBINE_MODE!r})
    return material

stage.DefinePrim("/World/droid_eval_physics", "Scope")
finger_material = define_physics_material(
    "/World/droid_eval_physics/FingerMaterial",
    {_DROID_FINGER_STATIC_FRICTION},
    {_DROID_FINGER_DYNAMIC_FRICTION},
)
cube_material = define_physics_material(
    "/World/droid_eval_physics/CubeMaterial",
    {_DROID_CUBE_STATIC_FRICTION},
    {_DROID_CUBE_DYNAMIC_FRICTION},
)
receptacle_material = define_physics_material(
    "/World/droid_eval_physics/ReceptacleMaterial",
    {_DROID_RECEPTACLE_STATIC_FRICTION},
    {_DROID_RECEPTACLE_DYNAMIC_FRICTION},
)
table_material = define_physics_material(
    "/World/droid_eval_physics/TableMaterial",
    {_DROID_TABLE_STATIC_FRICTION},
    {_DROID_TABLE_DYNAMIC_FRICTION},
)

joint_settings = json.loads({json.dumps(joint_settings)})
configured_joints = []
configured_gripper = False
gripper_joint_path = None
articulation_roots = 0
rigid_bodies = 0
finger_ccd_rigid_bodies = 0
for prim in Usd.PrimRange(root):
    name = prim.GetName()
    if name in joint_settings:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"DROID arm joint is not revolute: {{prim.GetPath()}}")
        drive = UsdPhysics.DriveAPI.Get(prim, UsdPhysics.Tokens.angular)
        if not drive or not drive.GetPrim().IsValid():
            raise RuntimeError(f"DROID arm drive unavailable: {{prim.GetPath()}}")
        settings = joint_settings[name]
        drive.GetStiffnessAttr().Set(400.0)
        drive.GetDampingAttr().Set(80.0)
        drive.GetMaxForceAttr().Set(settings["effort_limit"])
        joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
        joint_api.GetMaxJointVelocityAttr().Set(
            settings["velocity_limit_degrees"]
        )
        configured_joints.append(name)
    if name == {gripper_joint_name}:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"DROID gripper joint is not revolute: {{prim.GetPath()}}")
        PhysxSchema.PhysxJointAPI.Apply(prim).GetMaxJointVelocityAttr().Set(
            {gripper_velocity_limit_degrees}
        )
        configured_gripper = True
        gripper_joint_path = str(prim.GetPath())
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_api.GetDisableGravityAttr().Set(True)
        rigid_api.GetMaxDepenetrationVelocityAttr().Set(
            {_DROID_MAX_DEPENETRATION_VELOCITY_MPS}
        )
        prim_path = str(prim.GetPath())
        if any(
            prim_path == binding_path
            or prim_path.startswith(binding_path + "/")
            for binding_path in finger_binding_paths
        ):
            # Articulated fingers can cross a small cube between contact updates;
            # CCD keeps the physical solver aligned with the telemetry sweep.
            rigid_api.GetEnableCCDAttr().Set(True)
            finger_ccd_rigid_bodies += 1
        rigid_bodies += 1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        articulation_api.GetEnabledSelfCollisionsAttr().Set(False)
        articulation_api.GetSolverPositionIterationCountAttr().Set(
            {solver_position_iterations}
        )
        articulation_api.GetSolverVelocityIterationCountAttr().Set(
            {solver_velocity_iterations}
        )
        articulation_roots += 1

missing_joints = sorted(set(joint_settings) - set(configured_joints))
if missing_joints:
    raise RuntimeError(f"DROID dynamics joints missing: {{missing_joints}}")
if not configured_gripper:
    raise RuntimeError("DROID gripper dynamics joint missing")
if articulation_roots < 1 or rigid_bodies < 1:
    raise RuntimeError(
        "DROID dynamics profile found no articulation root or rigid bodies"
    )
if finger_ccd_rigid_bodies < 2:
    raise RuntimeError(
        "DROID dynamics profile did not enable CCD on both inner-finger bodies"
    )

physics_scenes = 0
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        physx_scene_api.GetTimeStepsPerSecondAttr().Set({physics_hz})
        physx_scene_api.GetEnableCCDAttr().Set(True)
        physics_scenes += 1
if physics_scenes < 1:
    raise RuntimeError("DROID dynamics profile found no physics scene")
physics_context = PhysicsContext(set_defaults=False)
physics_context.set_solver_type("TGS")
physics_context.set_solve_articulation_contact_last(True)

cube_root = stage.GetPrimAtPath(cube_path)
if not cube_root.IsValid():
    raise RuntimeError("DROID cube prim unavailable: %s" % cube_path)
# The scene payload omits mass, so pin the benchmark value before rebuilding PhysX.
UsdPhysics.MassAPI.Apply(cube_root).CreateMassAttr({_DROID_CUBE_MASS_KG})
cube_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_root)
cube_rigid_api.GetEnableCCDAttr().Set(True)
cube_rigid_api.GetMaxDepenetrationVelocityAttr().Set(
    {_DROID_MAX_DEPENETRATION_VELOCITY_MPS}
)
receptacle_root = stage.GetPrimAtPath(receptacle_path)
if not receptacle_root.IsValid():
    raise RuntimeError("DROID receptacle prim unavailable: %s" % receptacle_path)
if receptacle_root.HasAPI(UsdPhysics.RigidBodyAPI):
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        receptacle_root
    ).GetMaxDepenetrationVelocityAttr().Set(
        {_DROID_MAX_DEPENETRATION_VELOCITY_MPS}
    )
table_root = stage.GetPrimAtPath(table_path)
if not table_root.IsValid():
    raise RuntimeError("DROID table prim unavailable: %s" % table_path)

def configure_contact_root(path, material):
    profile_root = stage.GetPrimAtPath(path)
    if not profile_root.IsValid():
        raise RuntimeError("DROID contact root unavailable: %s" % path)
    UsdShade.MaterialBindingAPI(profile_root).Bind(
        material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )
    configured = 0
    for prim in Usd.PrimRange(profile_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision_api.CreateContactOffsetAttr({_DROID_CONTACT_OFFSET_METERS})
        collision_api.CreateRestOffsetAttr({_DROID_REST_OFFSET_METERS})
        configured += 1
    if configured < 1:
        raise RuntimeError("DROID contact root has no collision geometry: %s" % path)
    return configured

configured_finger_collisions = sum(
    configure_contact_root(path, finger_material)
    for path in finger_binding_paths
)
configured_cube_collisions = configure_contact_root(cube_path, cube_material)
configured_receptacle_collisions = configure_contact_root(
    receptacle_path, receptacle_material
)
configured_table_collisions = configure_contact_root(table_path, table_material)

app = omni.kit.app.get_app()
old_physics_view = SimulationManager.get_physics_sim_view()
if timeline.is_stopped():
    timeline.play()
    timeline.commit()
timeline.stop()
timeline.commit()
app.update()
if SimulationManager.get_physics_sim_view() is not None:
    raise RuntimeError("DROID hard stop did not invalidate the physics tensor view")

timeline.play()
timeline.commit()
app.update()
new_physics_view = SimulationManager.get_physics_sim_view()
if new_physics_view is None or new_physics_view is old_physics_view:
    raise RuntimeError("DROID physics tensor view was not rebuilt from final USD")
articulation_view = new_physics_view.create_articulation_view([robot_path])
if (
    articulation_view is None
    or not articulation_view.check()
    or articulation_view.count != 1
    or list(articulation_view.prim_paths) != [robot_path]
    or articulation_view.shared_metatype is None
):
    raise RuntimeError("DROID articulation metadata unavailable after physics rebuild")
timeline.pause()
timeline.commit()
if timeline.is_playing() or timeline.is_stopped():
    raise RuntimeError("DROID physics rebuild did not finish paused")

def read_float(prim, attribute_name, *, required=False):
    attribute = prim.GetAttribute(attribute_name)
    value = attribute.Get() if attribute and attribute.IsValid() else None
    if value is None:
        if required:
            raise RuntimeError(
                "DROID runtime attribute unavailable: %s.%s"
                % (prim.GetPath(), attribute_name)
            )
        return None
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(
            "DROID runtime attribute is not finite: %s.%s"
            % (prim.GetPath(), attribute_name)
        )
    return value

def read_offset_metadata(prim, attribute_name):
    attribute = prim.GetAttribute(attribute_name)
    value = attribute.Get() if attribute and attribute.IsValid() else None
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        raise RuntimeError(
            "DROID runtime offset is NaN: %s.%s"
            % (prim.GetPath(), attribute_name)
        )
    if math.isinf(value):
        return "schema:+inf" if value > 0 else "schema:-inf"
    return value

def material_profile_from_binding(binding_prim):
    binding_api = UsdShade.MaterialBindingAPI(binding_prim)
    for purpose in ("physics", UsdShade.Tokens.allPurpose):
        bound_material, _ = binding_api.ComputeBoundMaterial(purpose)
        material = bound_material.GetPrim() if bound_material else None
        if material is None or not material.IsValid():
            continue
        static_friction = read_float(material, "physics:staticFriction")
        dynamic_friction = read_float(material, "physics:dynamicFriction")
        if static_friction is None or dynamic_friction is None:
            continue
        combine_attribute = material.GetAttribute(
            "physxMaterial:frictionCombineMode"
        )
        combine_mode = (
            str(combine_attribute.Get())
            if combine_attribute
            and combine_attribute.IsValid()
            and combine_attribute.Get() is not None
            else None
        )
        return {{
            "path": str(material.GetPath()),
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "friction_combine_mode": combine_mode,
        }}
    return None

def collision_profile(profile_root):
    collision_count = 0
    contact_offsets = set()
    rest_offsets = set()
    entries = []
    for prim in Usd.PrimRange(profile_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_count += 1
        contact_offset = read_offset_metadata(
            prim,
            "physxCollision:contactOffset",
        )
        rest_offset = read_offset_metadata(
            prim,
            "physxCollision:restOffset",
        )
        if contact_offset is not None:
            contact_offsets.add(contact_offset)
        if rest_offset is not None:
            rest_offsets.add(rest_offset)
        approximation = prim.GetAttribute("physics:approximation")
        entries.append({{
            "prim_path": str(prim.GetPath()),
            "prim_type": prim.GetTypeName(),
            "approximation": (
                str(approximation.Get())
                if approximation
                and approximation.IsValid()
                and approximation.Get() is not None
                else None
            ),
            "contact_offset": contact_offset,
            "rest_offset": rest_offset,
        }})
    return {{
        "collision_count": collision_count,
        "contact_offsets": sorted(contact_offsets, key=str),
        "rest_offsets": sorted(rest_offsets, key=str),
        "entries": entries,
    }}

def require_contact_offsets(profile, label):
    for entry in profile["entries"]:
        if not math.isclose(
            entry["contact_offset"],
            {_DROID_CONTACT_OFFSET_METERS},
            rel_tol=1e-6,
            abs_tol=1e-9,
        ) or not math.isclose(
            entry["rest_offset"],
            {_DROID_REST_OFFSET_METERS},
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "DROID contact offsets mismatch for %s: %s" % (label, entry)
            )

def mass_profile(profile_root):
    entries = []
    for prim in Usd.PrimRange(profile_root):
        for attribute_name in ("physics:mass", "physics:density"):
            value = read_float(prim, attribute_name)
            if value is not None and value > 0:
                entries.append({{
                    "prim_path": str(prim.GetPath()),
                    "attribute": attribute_name,
                    "value": value,
                }})
    return entries

gripper_prim = stage.GetPrimAtPath(gripper_joint_path)
gripper_drive = UsdPhysics.DriveAPI.Get(
    gripper_prim,
    UsdPhysics.Tokens.angular,
)
if not gripper_drive or not gripper_drive.GetPrim().IsValid():
    raise RuntimeError("DROID runtime gripper drive unavailable")
gripper_profile = {{
    "joint_path": gripper_joint_path,
    "stiffness": float(gripper_drive.GetStiffnessAttr().Get()),
    "damping": float(gripper_drive.GetDampingAttr().Get()),
    "max_force": float(gripper_drive.GetMaxForceAttr().Get()),
    "max_joint_velocity_degrees": read_float(
        gripper_prim,
        "physxJoint:maxJointVelocity",
        required=True,
    ),
}}

expected_gripper = {{
    "stiffness": {_DROID_GRIPPER_STIFFNESS},
    "damping": {_DROID_GRIPPER_DAMPING},
    "max_force": {_DROID_GRIPPER_MAX_FORCE},
    "max_joint_velocity_degrees": {gripper_velocity_limit_degrees},
}}
for name, expected in expected_gripper.items():
    actual = gripper_profile[name]
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            "DROID runtime gripper mismatch for %s: expected=%s actual=%s"
            % (name, expected, actual)
        )

robot_collisions = collision_profile(root)
cube_collisions = collision_profile(cube_root)
cube_mass = mass_profile(cube_root)
if robot_collisions["collision_count"] < 1:
    raise RuntimeError("DROID robot collision profile is empty")
if cube_collisions["collision_count"] < 1:
    raise RuntimeError("DROID cube collision profile is empty")
if not cube_mass:
    raise RuntimeError("DROID cube has no positive authored mass or density")
authored_cube_mass = read_float(cube_root, "physics:mass", required=True)
if not math.isclose(
    authored_cube_mass,
    {_DROID_CUBE_MASS_KG},
    rel_tol=1e-6,
    abs_tol=1e-6,
):
    raise RuntimeError(
        "DROID cube mass mismatch: expected=%s actual=%s"
        % ({_DROID_CUBE_MASS_KG}, authored_cube_mass)
    )

finger_bindings = []
robot_materials = {{}}
bound_robot_collisions = 0
for binding_path in finger_binding_paths:
    binding_prim = stage.GetPrimAtPath(binding_path)
    if not binding_prim.IsValid():
        raise RuntimeError(
            "DROID finger physics binding prim unavailable: %s" % binding_path
        )
    material = material_profile_from_binding(binding_prim)
    if material is None:
        raise RuntimeError(
            "DROID finger physics material unavailable: %s" % binding_path
        )
    binding_collision_profile = collision_profile(binding_prim)
    binding_collision_count = binding_collision_profile["collision_count"]
    if binding_collision_count < 1:
        raise RuntimeError(
            "DROID finger physics binding has no collision geometry: %s"
            % binding_path
        )
    bound_robot_collisions += binding_collision_count
    require_contact_offsets(binding_collision_profile, binding_path)
    robot_materials[material["path"]] = material
    finger_bindings.append({{
        "binding_path": binding_path,
        "collision_count": binding_collision_count,
        "collision_profile": binding_collision_profile,
        "material_path": material["path"],
    }})
robot_collisions["bound_collision_count"] = bound_robot_collisions
robot_collisions["material_bindings"] = finger_bindings
robot_collisions["materials"] = [
    robot_materials[path] for path in sorted(robot_materials)
]

cube_material = material_profile_from_binding(cube_root)
if cube_material is None:
    raise RuntimeError("DROID cube physics material unavailable")
cube_collisions["bound_collision_count"] = cube_collisions["collision_count"]
cube_collisions["material_bindings"] = [{{
    "binding_path": cube_path,
    "collision_count": cube_collisions["collision_count"],
    "material_path": cube_material["path"],
}}]
cube_collisions["materials"] = [cube_material]
require_contact_offsets(cube_collisions, cube_path)

receptacle_collisions = collision_profile(receptacle_root)
receptacle_material_profile = material_profile_from_binding(receptacle_root)
if receptacle_material_profile is None:
    raise RuntimeError("DROID receptacle physics material unavailable")
receptacle_collisions["materials"] = [receptacle_material_profile]
require_contact_offsets(receptacle_collisions, receptacle_path)

table_collisions = collision_profile(table_root)
table_material_profile = material_profile_from_binding(table_root)
if table_material_profile is None:
    raise RuntimeError("DROID table physics material unavailable")
table_collisions["materials"] = [table_material_profile]
require_contact_offsets(table_collisions, table_path)

robot_materials = robot_collisions["materials"]
if len(robot_materials) != 1 or not all(
    math.isclose(
        material["static_friction"],
        {_DROID_FINGER_STATIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and math.isclose(
        material["dynamic_friction"],
        {_DROID_FINGER_DYNAMIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and material["friction_combine_mode"] == {_DROID_FRICTION_COMBINE_MODE!r}
    for material in robot_materials
):
    raise RuntimeError("DROID finger-pad physics material mismatch")
cube_materials = cube_collisions["materials"]
if len(cube_materials) != 1 or not all(
    math.isclose(
        material["static_friction"],
        {_DROID_CUBE_STATIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and math.isclose(
        material["dynamic_friction"],
        {_DROID_CUBE_DYNAMIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and material["friction_combine_mode"] == {_DROID_FRICTION_COMBINE_MODE!r}
    for material in cube_materials
):
    raise RuntimeError("DROID cube physics material mismatch")
if not (
    math.isclose(
        receptacle_material_profile["static_friction"],
        {_DROID_RECEPTACLE_STATIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and math.isclose(
        receptacle_material_profile["dynamic_friction"],
        {_DROID_RECEPTACLE_DYNAMIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and receptacle_material_profile["friction_combine_mode"]
    == {_DROID_FRICTION_COMBINE_MODE!r}
):
    raise RuntimeError("DROID receptacle physics material mismatch")
if not (
    math.isclose(
        table_material_profile["static_friction"],
        {_DROID_TABLE_STATIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and math.isclose(
        table_material_profile["dynamic_friction"],
        {_DROID_TABLE_DYNAMIC_FRICTION},
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    and table_material_profile["friction_combine_mode"]
    == {_DROID_FRICTION_COMBINE_MODE!r}
):
    raise RuntimeError("DROID table physics material mismatch")

runtime_profile = {{
    "status": "success",
    "profile": "{_DROID_DYNAMICS_PROFILE}",
    "physics_hz": {physics_hz},
    "fixed_time_stepping": True,
    "play_every_frame": True,
    "timeline_ticks_per_frame": 1,
    "configured_joints": sorted(configured_joints),
    "configured_gripper": configured_gripper,
    "articulation_roots": articulation_roots,
    "rigid_bodies": rigid_bodies,
    "finger_ccd_enabled": True,
    "finger_ccd_rigid_bodies": finger_ccd_rigid_bodies,
    "physics_scenes": physics_scenes,
    "solver_type": "TGS",
    "solver_position_iterations": {solver_position_iterations},
    "solver_velocity_iterations": {solver_velocity_iterations},
    "solve_articulation_contact_last": True,
    "cube_ccd_enabled": True,
    "contact_offset_meters": {_DROID_CONTACT_OFFSET_METERS},
    "rest_offset_meters": {_DROID_REST_OFFSET_METERS},
    "max_depenetration_velocity_mps": (
        {_DROID_MAX_DEPENETRATION_VELOCITY_MPS}
    ),
    "configured_finger_contact_collisions": configured_finger_collisions,
    "configured_cube_contact_collisions": configured_cube_collisions,
    "configured_receptacle_contact_collisions": (
        configured_receptacle_collisions
    ),
    "configured_table_contact_collisions": configured_table_collisions,
    "physics_context_reinitialized": True,
    "physics_view_replaced": True,
    "timeline_state": "paused",
    "gripper_drive": gripper_profile,
    "robot_collisions": robot_collisions,
    "cube_collisions": cube_collisions,
    "receptacle_collisions": receptacle_collisions,
    "table_collisions": table_collisions,
    "cube_mass": cube_mass,
}}
print(
    "{_DROID_DYNAMICS_STDOUT_PREFIX}"
    + json.dumps(runtime_profile, sort_keys=True)
)
"""
        result = self._call(mcp, "isaac.execute_script", {"code": code})
        stdout = result.get("stdout")
        if not isinstance(stdout, str):
            raise HostedDroidError(
                "isaac.execute_script did not return DROID dynamics stdout"
            )
        encoded_profile = next(
            (
                line.removeprefix(_DROID_DYNAMICS_STDOUT_PREFIX)
                for line in reversed(stdout.splitlines())
                if line.startswith(_DROID_DYNAMICS_STDOUT_PREFIX)
            ),
            None,
        )
        if encoded_profile is None:
            raise HostedDroidError(
                "isaac.execute_script did not emit the DROID dynamics marker"
            )
        try:
            profile = json.loads(encoded_profile)
        except json.JSONDecodeError as exc:
            raise HostedDroidError("Isaac emitted invalid DROID dynamics JSON") from exc
        if (
            not isinstance(profile, dict)
            or profile.get("status") != "success"
            or profile.get("profile") != _DROID_DYNAMICS_PROFILE
        ):
            raise HostedDroidError("Isaac emitted invalid DROID dynamics profile")
        observed_physics_hz = profile.get("physics_hz")
        if (
            isinstance(observed_physics_hz, bool)
            or not isinstance(observed_physics_hz, (int, float))
            or not math.isclose(
                float(observed_physics_hz),
                physics_hz,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise HostedDroidError("Isaac emitted the wrong DROID physics rate")
        for field_name, expected in (
            ("solver_position_iterations", solver_position_iterations),
            ("solver_velocity_iterations", solver_velocity_iterations),
        ):
            observed = profile.get(field_name)
            if isinstance(observed, bool) or observed != expected:
                raise HostedDroidError(f"Isaac emitted the wrong DROID {field_name}")
        if profile.get("finger_ccd_enabled") is not True:
            raise HostedDroidError(
                "Isaac emitted a DROID dynamics profile without inner-finger CCD"
            )
        finger_ccd_rigid_bodies = profile.get("finger_ccd_rigid_bodies")
        if (
            isinstance(finger_ccd_rigid_bodies, bool)
            or not isinstance(finger_ccd_rigid_bodies, int)
            or finger_ccd_rigid_bodies < 2
        ):
            raise HostedDroidError(
                "Isaac emitted an incomplete inner-finger CCD profile"
            )
        return cast(dict[str, Any], profile)

    def _reset_robot_for_policy(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
    ) -> str:
        arm_indices, gripper_index = joint_indices
        return self._set_joint_positions_runtime(
            mcp,
            joint_positions=[*_DROID_INITIAL_ARM_JOINT_POSITIONS, 0.0],
            joint_indices=[*arm_indices, gripper_index],
        )

    def _set_joint_positions_runtime(
        self,
        mcp: MCPClient,
        *,
        joint_positions: list[float],
        joint_indices: list[int],
    ) -> str:
        result = self._call(
            mcp,
            "isaac.set_joint_positions",
            {
                "prim_path": self.config.robot_prim_path,
                "joint_positions": joint_positions,
                "joint_indices": joint_indices,
                "require_runtime": True,
            },
        )
        control_source = result.get("control_source")
        if control_source != "runtime_articulation":
            raise HostedDroidError(
                "isaac.set_joint_positions did not prove runtime articulation "
                f"control: {control_source!r}"
            )
        return control_source

    def _settle_robot_for_policy(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        physics_steps_per_action: int,
    ) -> tuple[np.ndarray, float]:
        target = np.asarray(_DROID_INITIAL_ARM_JOINT_POSITIONS, dtype=np.float32)
        self._step_while_playing(
            mcp,
            num_steps=(physics_steps_per_action * _CAMERA_WARMUP_ACTION_PERIODS),
        )
        self._sleep(_CAMERA_WARMUP_SECONDS)

        previous: np.ndarray | None = None
        stable_checks = 0
        last_error = math.inf
        last_motion = math.inf
        last_gripper = math.inf
        for check in range(_INITIAL_SETTLE_MAX_CHECKS):
            arm_positions, gripper_position = self._joint_state(mcp, joint_indices)
            last_error = float(np.max(np.abs(arm_positions - target)))
            last_motion = (
                float(np.max(np.abs(arm_positions - previous)))
                if previous is not None
                else math.inf
            )
            last_gripper = gripper_position
            if (
                last_error <= _INITIAL_ARM_ERROR_TOLERANCE_RADIANS
                and last_motion <= _INITIAL_ARM_MOTION_TOLERANCE_RADIANS
                and gripper_position <= _INITIAL_GRIPPER_TOLERANCE
            ):
                stable_checks += 1
            else:
                stable_checks = 0
            if stable_checks >= _INITIAL_SETTLE_STABLE_CHECKS:
                return arm_positions, gripper_position
            previous = arm_positions.copy()
            if check + 1 < _INITIAL_SETTLE_MAX_CHECKS:
                self._step_while_playing(
                    mcp,
                    num_steps=physics_steps_per_action,
                )

        raise HostedDroidError(
            "DROID robot did not settle at the measured benchmark initial state: "
            f"arm_error={last_error:.6f}, arm_motion={last_motion:.6f}, "
            f"gripper={last_gripper:.6f}"
        )

    def _ensure_cameras(self, mcp: MCPClient) -> list[str]:
        created: list[str] = []
        for camera in self.config.cameras:
            self._create_camera(
                mcp,
                camera,
                resolution=(self.config.image_width, self.config.image_height),
            )
            created.append(camera.prim_path)
        return created

    def _ensure_viewer_camera(self, mcp: MCPClient) -> CameraSpec:
        viewer_camera = _viewer_camera_for(self.config.cameras[0])
        self._create_camera(
            mcp,
            viewer_camera,
            resolution=_VIEWER_CAMERA_RESOLUTION,
        )
        return viewer_camera

    def _create_camera(
        self,
        mcp: MCPClient,
        camera: CameraSpec,
        *,
        resolution: tuple[int, int],
    ) -> None:
        arguments = {
            "prim_path": camera.prim_path,
            "position": list(camera.position),
            "orientation": list(camera.orientation_wxyz),
            "resolution": list(resolution),
            "focal_length": camera.focal_length,
            "clipping_range": list(camera.clipping_range),
            "focus_distance": camera.focus_distance,
            "horizontal_aperture": camera.horizontal_aperture,
            "vertical_aperture": camera.vertical_aperture,
        }
        try:
            self._call(mcp, "isaac.create_camera", arguments)
        except HostedDroidError as exc:
            if not _is_legacy_camera_contract_failure(exc):
                raise
        else:
            self._configure_camera_image_contract(mcp, camera)
            return
        self._configure_legacy_camera(mcp, camera)
        self._call(
            mcp,
            "isaac.create_camera",
            {
                "prim_path": camera.prim_path,
                "resolution": list(resolution),
            },
        )

    def _configure_camera_image_contract(
        self,
        mcp: MCPClient,
        camera: CameraSpec,
    ) -> None:
        prim_path = json.dumps(camera.prim_path)
        projection = json.dumps(camera.projection)
        code = f"""
import omni.usd
from pxr import UsdGeom, Vt

stage = omni.usd.get_context().get_stage()
camera = UsdGeom.Camera.Get(stage, {prim_path})
if not camera or not camera.GetPrim().IsValid():
    raise RuntimeError("camera prim is missing after creation")
camera.GetProjectionAttr().Set({projection})
camera.GetHorizontalApertureOffsetAttr().Set({camera.horizontal_aperture_offset})
camera.GetVerticalApertureOffsetAttr().Set({camera.vertical_aperture_offset})
camera.GetFStopAttr().Set({camera.f_stop})
camera.GetClippingPlanesAttr().Set(Vt.Vec4fArray())
print({{"status": "success", "prim_path": {prim_path}}})
"""
        self._call(mcp, "isaac.execute_script", {"code": code})

    def _retire_previous_cameras(self, mcp: MCPClient) -> None:
        current_paths = {camera.prim_path for camera in self.config.cameras}
        current_external_roots = {
            path.rsplit("/", 1)[0]
            for path in current_paths
            if path.startswith(_DROID_EXTERNAL_CAMERA_ROOT_PREFIX)
        }
        stale_paths: set[str] = set()

        world_prims = self._listed_prim_paths(mcp, "/World")
        stale_paths.update(
            path
            for path in world_prims
            if path.startswith(_DROID_EXTERNAL_CAMERA_ROOT_PREFIX)
            and path not in current_external_roots
        )

        wrist_parent = "/World/robot/Gripper/Robotiq_2F_85/base_link"
        wrist_prefix = f"{wrist_parent}/{_DROID_WRIST_CAMERA_PREFIX}"
        wrist_prims = self._listed_prim_paths(mcp, wrist_parent)
        stale_paths.update(
            path
            for path in wrist_prims
            if path.startswith(wrist_prefix) and path not in current_paths
        )

        for path in sorted(stale_paths, key=lambda item: (-item.count("/"), item)):
            self._call(mcp, "isaac.delete_object", {"prim_path": path})

    def _listed_prim_paths(self, mcp: MCPClient, root_path: str) -> set[str]:
        payload = self._call(mcp, "isaac.list_prims", {"root_path": root_path})
        prims = payload.get("prims")
        if not isinstance(prims, list):
            raise HostedDroidError("isaac.list_prims did not return prims")
        paths = {
            prim.get("path")
            for prim in prims
            if isinstance(prim, Mapping) and isinstance(prim.get("path"), str)
        }
        if len(paths) != len(prims):
            raise HostedDroidError("isaac.list_prims returned malformed prim records")
        return cast(set[str], paths)

    def _set_viewer_camera(self, mcp: MCPClient, prim_path: str) -> None:
        if (
            self._try_call(
                mcp,
                "isaac.set_active_camera",
                {"prim_path": prim_path},
            )
            is not None
        ):
            return
        encoded_path = json.dumps(prim_path)
        code = f"""
import omni.kit.app
from omni.kit.viewport.utility import get_active_viewport

viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("no active viewport")
viewport.camera_path = {encoded_path}
omni.kit.app.get_app().update()
print({{"status": "success", "active_camera": str(viewport.camera_path)}})
"""
        self._call(mcp, "isaac.execute_script", {"code": code})

    def _validate_policy_cameras(self, mcp: MCPClient) -> None:
        expected = [
            {
                "role": camera.role,
                "prim_path": camera.prim_path,
                "transform_space": (
                    "world" if camera.role.startswith("exterior_") else "local"
                ),
                "position": list(camera.position),
                "orientation_wxyz": list(camera.orientation_wxyz),
                "focal_length": camera.focal_length,
                "clipping_range": list(camera.clipping_range),
                "focus_distance": camera.focus_distance,
                "horizontal_aperture": camera.horizontal_aperture,
                "vertical_aperture": camera.vertical_aperture,
                "projection": camera.projection,
                "horizontal_aperture_offset": camera.horizontal_aperture_offset,
                "vertical_aperture_offset": camera.vertical_aperture_offset,
                "f_stop": camera.f_stop,
            }
            for camera in self.config.cameras
        ]
        prefix = json.dumps(_CAMERA_CALIBRATION_STDOUT_PREFIX)
        code = f"""
import json
import math
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
expected = {json.dumps(expected, sort_keys=True)}
position_tolerance = {_CAMERA_POSITION_TOLERANCE_METERS}
orientation_tolerance = {_CAMERA_ORIENTATION_ALIGNMENT_TOLERANCE}
optics_tolerance = {_CAMERA_OPTICS_TOLERANCE}
camera_results = []
xform_cache = UsdGeom.XformCache()

for spec in expected:
    path = spec["prim_path"]
    try:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError("camera prim is missing or has the wrong type")
        if spec["transform_space"] == "world":
            matrix = xform_cache.GetLocalToWorldTransform(prim)
        else:
            matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        transform = Gf.Transform(matrix)
        translation = transform.GetTranslation()
        scale = transform.GetScale()
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        actual_orientation = [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ]
        expected_orientation = spec["orientation_wxyz"]
        actual_norm = math.sqrt(sum(value * value for value in actual_orientation))
        expected_norm = math.sqrt(
            sum(value * value for value in expected_orientation)
        )
        if (
            not math.isfinite(actual_norm)
            or not math.isfinite(expected_norm)
            or actual_norm <= 1e-12
            or expected_norm <= 1e-12
        ):
            raise RuntimeError("camera orientation is non-finite or zero")
        orientation_alignment = abs(
            sum(
                actual * wanted
                for actual, wanted in zip(
                    actual_orientation,
                    expected_orientation,
                )
            )
            / (actual_norm * expected_norm)
        )
        position_error = math.sqrt(
            sum(
                (float(translation[index]) - spec["position"][index]) ** 2
                for index in range(3)
            )
        )

        camera = UsdGeom.Camera(prim)
        clipping = camera.GetClippingRangeAttr().Get()
        actual_optics = {{
            "focal_length": float(camera.GetFocalLengthAttr().Get()),
            "focus_distance": float(camera.GetFocusDistanceAttr().Get()),
            "horizontal_aperture": float(
                camera.GetHorizontalApertureAttr().Get()
            ),
            "vertical_aperture": float(camera.GetVerticalApertureAttr().Get()),
            "horizontal_aperture_offset": float(
                camera.GetHorizontalApertureOffsetAttr().Get()
            ),
            "vertical_aperture_offset": float(
                camera.GetVerticalApertureOffsetAttr().Get()
            ),
            "f_stop": float(camera.GetFStopAttr().Get()),
            "clipping_range": [float(clipping[0]), float(clipping[1])],
        }}
        expected_optics = {{
            name: spec[name]
            for name in (
                "focal_length",
                "focus_distance",
                "horizontal_aperture",
                "vertical_aperture",
                "horizontal_aperture_offset",
                "vertical_aperture_offset",
                "f_stop",
                "clipping_range",
            )
        }}
        optics_error_by_field = {{
            name: abs(actual_optics[name] - expected_optics[name])
            for name in (
                "focal_length",
                "focus_distance",
                "horizontal_aperture",
                "vertical_aperture",
                "horizontal_aperture_offset",
                "vertical_aperture_offset",
                "f_stop",
            )
        }}
        optics_error_by_field["clipping_range_near"] = abs(
            actual_optics["clipping_range"][0] - expected_optics["clipping_range"][0]
        )
        optics_error_by_field["clipping_range_far"] = abs(
            actual_optics["clipping_range"][1] - expected_optics["clipping_range"][1]
        )
        maximum_optics_error = max(optics_error_by_field.values())
        scale_error = max(abs(float(value) - 1.0) for value in scale)
        projection = str(camera.GetProjectionAttr().Get())
        clipping_planes = camera.GetClippingPlanesAttr().Get()
        clipping_plane_count = len(clipping_planes) if clipping_planes is not None else 0
        metrics = (
            position_error,
            orientation_alignment,
            maximum_optics_error,
            scale_error,
        )
        issues = []
        if not all(math.isfinite(value) for value in metrics):
            issues.append("non_finite")
        if position_error > position_tolerance:
            issues.append("position")
        if 1.0 - min(1.0, orientation_alignment) > orientation_tolerance:
            issues.append("orientation")
        if maximum_optics_error > optics_tolerance:
            issues.append("optics")
        if scale_error > optics_tolerance:
            issues.append("scale")
        if projection != spec["projection"]:
            issues.append("projection")
        if clipping_plane_count != 0:
            issues.append("clipping_planes")
        camera_results.append({{
            "role": spec["role"],
            "prim_path": path,
            "transform_space": spec["transform_space"],
            "valid": not issues,
            "issues": issues,
            "position_error_meters": position_error,
            "orientation_alignment": orientation_alignment,
            "maximum_optics_error": maximum_optics_error,
            "optics_error_by_field": optics_error_by_field,
            "actual_optics": actual_optics,
            "expected_optics": expected_optics,
            "scale_error": scale_error,
            "projection": projection,
            "clipping_plane_count": clipping_plane_count,
        }})
    except Exception as error:
        camera_results.append({{
            "prim_path": path,
            "valid": False,
            "issues": ["unreadable"],
            "error": str(error),
        }})

payload = {{
    "schema_version": 2,
    "valid": all(result["valid"] for result in camera_results),
    "cameras": camera_results,
}}
print({prefix} + json.dumps(payload, sort_keys=True))
"""
        response = self._call(mcp, "isaac.execute_script", {"code": code})
        stdout = response.get("stdout")
        if not isinstance(stdout, str):
            raise HostedDroidError(
                "isaac.execute_script did not return camera calibration stdout"
            )
        encoded = next(
            (
                line.removeprefix(_CAMERA_CALIBRATION_STDOUT_PREFIX)
                for line in reversed(stdout.splitlines())
                if line.startswith(_CAMERA_CALIBRATION_STDOUT_PREFIX)
            ),
            None,
        )
        if encoded is None:
            raise HostedDroidError(
                "isaac.execute_script did not emit the camera calibration marker"
            )
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise HostedDroidError(
                "Isaac emitted invalid camera calibration JSON"
            ) from exc
        cameras = payload.get("cameras") if isinstance(payload, Mapping) else None
        expected_contract = {
            (
                camera.prim_path,
                camera.role,
                "world" if camera.role.startswith("exterior_") else "local",
            )
            for camera in self.config.cameras
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 2
            or not isinstance(cameras, list)
            or len(cameras) != len(expected_contract)
            or not all(isinstance(camera, Mapping) for camera in cameras)
            or {
                (
                    camera.get("prim_path"),
                    camera.get("role"),
                    camera.get("transform_space"),
                )
                for camera in cameras
            }
            != expected_contract
        ):
            raise HostedDroidError("Isaac emitted an incomplete camera calibration")
        invalid = [camera for camera in cameras if camera.get("valid") is not True]
        if payload.get("valid") is not True or invalid:
            details = "; ".join(
                f"{camera.get('prim_path')}: "
                + json.dumps(
                    {
                        "issues": camera.get("issues"),
                        "maximum_optics_error": camera.get("maximum_optics_error"),
                        "optics_error_by_field": camera.get("optics_error_by_field"),
                        "actual_optics": camera.get("actual_optics"),
                        "expected_optics": camera.get("expected_optics"),
                    },
                    sort_keys=True,
                )
                for camera in invalid
            )
            raise HostedDroidError(
                "DROID policy camera calibration drifted before sampling: " + details
            )

    def _configure_legacy_camera(self, mcp: MCPClient, camera: CameraSpec) -> None:
        position = json.dumps(list(camera.position))
        orientation = json.dumps(list(camera.orientation_wxyz))
        prim_path = json.dumps(camera.prim_path)
        projection = json.dumps(camera.projection)
        code = f"""
import omni.usd
from pxr import Gf, UsdGeom, Vt

stage = omni.usd.get_context().get_stage()
camera = UsdGeom.Camera.Define(stage, {prim_path})
prim = camera.GetPrim()
xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
position = {position}
orientation = {orientation}
xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Vec3d(*position)
)
xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
)
camera.GetFocalLengthAttr().Set({camera.focal_length})
camera.GetClippingRangeAttr().Set(
    Gf.Vec2f({camera.clipping_range[0]}, {camera.clipping_range[1]})
)
camera.GetFocusDistanceAttr().Set({camera.focus_distance})
camera.GetHorizontalApertureAttr().Set({camera.horizontal_aperture})
camera.GetVerticalApertureAttr().Set({camera.vertical_aperture})
camera.GetProjectionAttr().Set({projection})
camera.GetHorizontalApertureOffsetAttr().Set({camera.horizontal_aperture_offset})
camera.GetVerticalApertureOffsetAttr().Set({camera.vertical_aperture_offset})
camera.GetFStopAttr().Set({camera.f_stop})
camera.GetClippingPlanesAttr().Set(Vt.Vec4fArray())
print({{"status": "success", "prim_path": {prim_path}}})
"""
        self._call(mcp, "isaac.execute_script", {"code": code})

    def _joint_indices(self, mcp: MCPClient) -> tuple[list[int], int]:
        info = self._call(
            mcp,
            "isaac.get_robot_info",
            {
                "prim_path": self.config.robot_prim_path,
                "require_runtime": True,
                "refresh_runtime": True,
            },
        )
        if info.get("measurement_source") != "runtime_articulation":
            raise HostedDroidError(
                "isaac.get_robot_info did not prove runtime articulation ordering"
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
        physics_steps_per_action: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> _RolloutOutcome:
        samples = 0
        action_steps = 0
        tracker: _DroidTaskSuccessTracker | None = None
        terminal_failure_reason: str | None = None
        if self.config.task_success is not None:
            initial_state = self._capture_task_state(mcp, self.config.task_success)
            replay_task_comparison = self._validate_replay_initial_task_state(
                initial_state
            )
            tracker = _DroidTaskSuccessTracker(
                self.config.task_success,
                initial_state,
            )
            if evidence is not None:
                initial_evaluation = tracker.initial_evaluation()
                if replay_task_comparison is not None:
                    initial_evaluation["replay_initial_task_state"] = (
                        replay_task_comparison
                    )
                evidence.write_task_state(
                    phase="initial",
                    action_index=None,
                    state=initial_state,
                    evaluation=initial_evaluation,
                )
        should_stop = False
        while action_steps < self.config.max_action_steps:
            observation = self._observation(mcp, joint_indices, samples, evidence)
            response = self.sampling_api.sample_droid(
                observation,
                timeout=self.config.request_timeout_seconds,
            )
            sampled_chunk = _action_chunk(response)
            _validate_policy_response(self.config.base_model, response, sampled_chunk)
            chunk = sampled_chunk[: self.config.open_loop_horizon]
            sample_index = samples
            if evidence is not None:
                evidence.write_sample(
                    sample_index,
                    response,
                    sampled_chunk,
                    chunk,
                )
            samples += 1
            for chunk_index, action in enumerate(chunk):
                if action_steps >= self.config.max_action_steps:
                    break
                on_target_accepted: Callable[[list[float], list[int]], None] | None = (
                    None
                )
                if evidence is not None:
                    on_target_accepted = partial(
                        evidence.write_action_target,
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        action_index=action_steps,
                        policy_action=action,
                    )
                (
                    joint_positions,
                    applied_joint_indices,
                    simulation_timing,
                ) = self._apply_action(
                    mcp,
                    joint_indices,
                    action,
                    physics_steps_per_action,
                    on_target_accepted=on_target_accepted,
                )
                if evidence is not None:
                    evidence.write_applied_action(
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        action_index=action_steps,
                        policy_action=action,
                        joint_positions=joint_positions,
                        joint_indices=applied_joint_indices,
                        simulation_timing=simulation_timing,
                    )
                    if self.config.record_video:
                        video_rgb = self._capture_rgb(
                            mcp,
                            self.config.cameras[0].prim_path,
                            action_steps,
                            0,
                            None,
                        )
                        evidence.write_video_frame(action_steps, video_rgb)
                if tracker is not None:
                    _, observed_gripper_position = self._joint_state(
                        mcp,
                        joint_indices,
                    )
                    task_state = self._capture_task_state(mcp, tracker.spec)
                    task_evaluation = tracker.evaluate(
                        task_state,
                        action_index=action_steps,
                        observed_gripper_position=observed_gripper_position,
                        commanded_gripper_closed=joint_positions[-1] > 0,
                        contact_integrity=cast(
                            Mapping[str, Any],
                            simulation_timing.get("contact_integrity"),
                        ),
                        expected_contact_updates=cast(
                            int,
                            simulation_timing["stepped"],
                        ),
                        expected_contact_physics_dt_seconds=cast(
                            float,
                            simulation_timing["before"]["physics_dt"],
                        ),
                    )
                    if evidence is not None:
                        evidence.write_task_state(
                            phase="post_action",
                            action_index=action_steps,
                            state=task_state,
                            evaluation=task_evaluation,
                        )
                    terminal_failure_reason = tracker.terminal_failure_reason()
                    should_stop = bool(task_evaluation["success"]) or (
                        terminal_failure_reason is not None
                    )
                action_steps += 1
                if should_stop:
                    break
            if should_stop:
                break
        if tracker is None:
            return _RolloutOutcome(
                samples=samples,
                action_steps=action_steps,
                task_success=None,
                task_success_predicate=None,
                task_success_action_index=None,
                task_success_checks=0,
                task_success_reason=None,
            )
        task_success = tracker.success_action_index is not None
        return _RolloutOutcome(
            samples=samples,
            action_steps=action_steps,
            task_success=task_success,
            task_success_predicate=tracker.spec.name,
            task_success_action_index=tracker.success_action_index,
            task_success_checks=tracker.checks,
            task_success_reason=(
                "physically_credible_policy_placement_proven"
                if task_success
                else terminal_failure_reason
                or "max_action_steps_reached_without_valid_placement"
            ),
        )

    def _capture_task_state(
        self,
        mcp: MCPClient,
        spec: DroidTaskSuccessSpec,
    ) -> _DroidTaskState:
        object_path = json.dumps(spec.object_prim_path)
        receptacle_path = json.dumps(spec.receptacle_prim_path)
        gripper_reference_path = json.dumps(spec.gripper_reference_prim_path)
        prefix = json.dumps(_TASK_STATE_STDOUT_PREFIX)
        code = f"""
import json
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_],
    useExtentsHint=True,
)

def read_bounds(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"task prim not found: {{path}}")
    bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return {{
        "minimum": [float(minimum[i]) for i in range(3)],
        "maximum": [float(maximum[i]) for i in range(3)],
    }}

def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)

def read_physics_tensor_states(paths):
    from isaacsim.core.simulation_manager import SimulationManager

    simulation_view = SimulationManager.get_physics_simulation_view()
    if simulation_view is None:
        raise RuntimeError("physics tensor simulation view is unavailable")
    states = {{}}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"rigid-body API unavailable: {{path}}")
        body_view = simulation_view.create_rigid_body_view(path)
        if body_view is None or not body_view.check():
            raise RuntimeError(f"physics tensor rigid-body view invalid: {{path}}")
        matched_paths = list(body_view.prim_paths)
        if body_view.count != 1 or matched_paths != [path]:
            raise RuntimeError(
                "physics tensor rigid-body view did not match the exact prim; "
                f"requested={{path}}; matched={{matched_paths}}"
            )
        values = as_numpy(body_view.get_velocities()).reshape(body_view.count, 6)[0]
        transform = as_numpy(body_view.get_transforms()).reshape(body_view.count, 7)[0]
        states[path] = {{
            "position": [float(transform[i]) for i in range(3)],
            "linear": [float(values[i]) for i in range(3)],
            "angular": [float(values[i]) for i in range(3, 6)],
        }}
    return states

def read_legacy_dynamic_control_states(paths):
    from omni.isaac.dynamic_control import _dynamic_control

    dynamic_control = _dynamic_control.acquire_dynamic_control_interface()
    states = {{}}
    for path in paths:
        handle = dynamic_control.get_rigid_body(path)
        if handle == _dynamic_control.INVALID_HANDLE:
            raise RuntimeError(f"legacy rigid-body handle unavailable: {{path}}")
        linear = dynamic_control.get_rigid_body_linear_velocity(handle)
        angular = dynamic_control.get_rigid_body_angular_velocity(handle)
        pose = dynamic_control.get_rigid_body_pose(handle)
        states[path] = {{
            "position": [float(pose.p[i]) for i in range(3)],
            "linear": [float(linear[i]) for i in range(3)],
            "angular": [float(angular[i]) for i in range(3)],
        }}
    return states

def read_states(paths):
    try:
        return read_physics_tensor_states(paths), "physics_tensor"
    except Exception as tensor_error:
        try:
            return read_legacy_dynamic_control_states(paths), "legacy_dynamic_control"
        except Exception as legacy_error:
            raise RuntimeError(
                "rigid-body state unavailable; "
                f"physics_tensor={{tensor_error}}; "
                f"legacy_dynamic_control={{legacy_error}}"
            ) from legacy_error

object_path = {object_path}
receptacle_path = {receptacle_path}
gripper_reference_path = {gripper_reference_path}
states, velocity_source = read_states(
    [object_path, receptacle_path, gripper_reference_path]
)
payload = {{
    "object_path": object_path,
    "receptacle_path": receptacle_path,
    "velocity_source": velocity_source,
    "object_bounds": read_bounds(object_path),
    "receptacle_bounds": read_bounds(receptacle_path),
    "object_velocity": states[object_path],
    "receptacle_velocity": states[receptacle_path],
    "object_runtime_position": states[object_path]["position"],
    "gripper_reference_path": gripper_reference_path,
    "gripper_reference_position": states[gripper_reference_path]["position"],
}}
print({prefix} + json.dumps(payload, sort_keys=True))
"""
        response = self._call(mcp, "isaac.execute_script", {"code": code})
        stdout = response.get("stdout")
        if not isinstance(stdout, str):
            raise HostedDroidError(
                "isaac.execute_script did not return task-state stdout"
            )
        encoded_state = next(
            (
                line.removeprefix(_TASK_STATE_STDOUT_PREFIX)
                for line in reversed(stdout.splitlines())
                if line.startswith(_TASK_STATE_STDOUT_PREFIX)
            ),
            None,
        )
        if encoded_state is None:
            raise HostedDroidError(
                "isaac.execute_script did not emit the task-state marker"
            )
        try:
            payload = json.loads(encoded_state)
        except json.JSONDecodeError as exc:
            raise HostedDroidError("Isaac emitted invalid task-state JSON") from exc
        return _parse_task_state(payload, spec)

    def _observation(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        sample_index: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> DroidObservation:
        self._validate_policy_cameras(mcp)
        images = [
            self._capture_rgb(
                mcp,
                camera.prim_path,
                sample_index,
                camera_index,
                evidence,
            )
            for camera_index, camera in enumerate(self.config.cameras)
        ]
        self._validate_policy_cameras(mcp)
        arm_positions, gripper = self._joint_state(mcp, joint_indices)
        return DroidObservation(
            exterior_image_1_left=images[0],
            exterior_image_2_left=images[1],
            wrist_image_left=images[2],
            joint_position=arm_positions,
            gripper_position=np.asarray([gripper], dtype=np.float32),
            instruction=self.config.instruction,
        )

    def _joint_state(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
    ) -> tuple[np.ndarray, float]:
        positions_payload = self._call(
            mcp,
            "isaac.get_joint_positions",
            {
                "prim_path": self.config.robot_prim_path,
                "require_runtime": True,
            },
        )
        if positions_payload.get("measurement_source") != "runtime_articulation":
            raise HostedDroidError(
                "isaac.get_joint_positions did not return runtime articulation state"
            )
        positions = np.asarray(
            positions_payload.get("joint_positions"),
            dtype=np.float32,
        )
        arm_indices, gripper_index = joint_indices
        if positions.ndim != 1 or positions.size <= max(*arm_indices, gripper_index):
            raise HostedDroidError(
                "isaac.get_joint_positions returned an incomplete joint vector"
            )
        gripper = float(
            np.clip(
                positions[gripper_index] / GRIPPER_CLOSED_RADIANS,
                0.0,
                1.0,
            )
        )
        return np.ascontiguousarray(positions[arm_indices]), gripper

    def _capture_rgb(
        self,
        mcp: MCPClient,
        camera_prim_path: str,
        sample_index: int,
        camera_index: int,
        evidence: _EvidenceRecorder | None = None,
    ) -> np.ndarray:
        output_path = (
            f"/data/workspace/media/droid-{sample_index:05d}-{camera_index}.png"
        )
        last_error: HostedDroidError | None = None
        for attempt in range(1, _CAMERA_CAPTURE_ATTEMPTS + 1):
            try:
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
                raw, rgb = _decode_valid_rgb(
                    encoded,
                    camera_prim_path=camera_prim_path,
                    expected_width=self.config.image_width,
                    expected_height=self.config.image_height,
                )
                if evidence is not None:
                    evidence.write_frame(sample_index, camera_index, raw)
                return rgb
            except HostedDroidError as exc:
                last_error = exc
                if attempt == _CAMERA_CAPTURE_ATTEMPTS:
                    break
                self._sleep(_CAMERA_CAPTURE_RETRY_SECONDS)
        raise HostedDroidError(
            f"camera {camera_prim_path} did not produce a valid rendered frame after "
            f"{_CAMERA_CAPTURE_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _control_cadence(self, mcp: MCPClient) -> tuple[float, int]:
        state = self._simulation_state(mcp)
        physics_dt = state["physics_dt"]
        expected_physics_dt = 1.0 / self.config.physics_hz
        if not math.isclose(
            physics_dt,
            expected_physics_dt,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise HostedDroidError(
                "Isaac runtime physics dt does not match the configured DROID rate: "
                f"expected={expected_physics_dt:.12f}s actual={physics_dt:.12f}s"
            )
        configured_steps = self.config.physics_steps_per_action
        if configured_steps is not None:
            return physics_dt, configured_steps
        steps = max(1, round(1.0 / (self.config.target_control_hz * physics_dt)))
        return physics_dt, steps

    def _simulation_state(self, mcp: MCPClient) -> dict[str, Any]:
        state = self._call(mcp, "isaac.get_simulation_state", {})
        physics_dt = state.get("physics_dt")
        current_time = state.get("current_time")
        timeline_state = state.get("timeline_state")
        if (
            isinstance(physics_dt, bool)
            or not isinstance(physics_dt, (int, float))
            or not math.isfinite(float(physics_dt))
            or float(physics_dt) <= 0
        ):
            raise HostedDroidError(
                f"isaac.get_simulation_state returned invalid physics_dt: {physics_dt!r}"
            )
        if (
            isinstance(current_time, bool)
            or not isinstance(current_time, (int, float))
            or not math.isfinite(float(current_time))
        ):
            raise HostedDroidError(
                "isaac.get_simulation_state returned invalid current_time: "
                f"{current_time!r}"
            )
        if timeline_state not in {"playing", "paused", "stopped"}:
            raise HostedDroidError(
                "isaac.get_simulation_state returned invalid timeline_state: "
                f"{timeline_state!r}"
            )
        return {
            "physics_dt": float(physics_dt),
            "current_time": float(current_time),
            "timeline_state": timeline_state,
        }

    def _step_while_playing(
        self,
        mcp: MCPClient,
        *,
        num_steps: int,
        observe_joints: list[str] | None = None,
        observe_cap: int | None = None,
        contact_integrity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            arguments: dict[str, Any] = {
                "num_steps": num_steps,
                "pause_after": True,
            }
            if observe_joints is not None:
                arguments["observe_joints"] = observe_joints
            if observe_cap is not None:
                arguments["observe_cap"] = observe_cap
            if contact_integrity is not None:
                arguments["contact_integrity"] = dict(contact_integrity)
            result = self._call(mcp, "isaac.step_simulation", arguments)
        except BaseException as step_exc:
            try:
                self._call(mcp, "isaac.pause_simulation", {})
            except Exception as pause_exc:
                raise HostedDroidError(
                    f"simulation step failed ({step_exc}); pause also failed: {pause_exc}"
                ) from step_exc
            raise
        state = self._simulation_state(mcp)
        if state["timeline_state"] != "paused":
            raise HostedDroidError(
                "isaac.step_simulation did not leave the timeline paused"
            )
        return result

    def _apply_action(
        self,
        mcp: MCPClient,
        joint_indices: tuple[list[int], int],
        action: np.ndarray,
        physics_steps_per_action: int,
        *,
        on_target_accepted: Callable[[list[float], list[int]], None] | None = None,
    ) -> tuple[list[float], list[int], dict[str, Any]]:
        arm_indices, gripper_index = joint_indices
        applied_joint_indices = [*arm_indices, gripper_index]
        joint_positions = _droid_joint_positions_for_policy_action(action)
        control_source = self._set_joint_positions_runtime(
            mcp,
            joint_positions=joint_positions,
            joint_indices=applied_joint_indices,
        )
        if on_target_accepted is not None:
            on_target_accepted(joint_positions, applied_joint_indices)
        before = self._simulation_state(mcp)
        step = self._step_while_playing(
            mcp,
            num_steps=physics_steps_per_action,
            observe_joints=[self.config.robot_prim_path],
            observe_cap=1,
            contact_integrity=(
                _contact_integrity_request(self.config.task_success)
                if self.config.task_success is not None
                else None
            ),
        )
        after = self._simulation_state(mcp)
        stepped = step.get("stepped")
        if (
            isinstance(stepped, bool)
            or not isinstance(stepped, int)
            or stepped != physics_steps_per_action
            or step.get("timed_out") is True
        ):
            raise HostedDroidError(
                "isaac.step_simulation applied an incomplete action: "
                f"expected {physics_steps_per_action} frames, "
                f"stepped={stepped!r}, timed_out={step.get('timed_out')!r}"
            )
        observed_seconds = after["current_time"] - before["current_time"]
        if observed_seconds < 0:
            raise HostedDroidError(
                "simulation time moved backward while applying action"
            )
        expected_seconds = physics_steps_per_action * before["physics_dt"]
        drift_seconds = observed_seconds - expected_seconds
        tolerance_seconds = max(1e-6, before["physics_dt"] * 0.25)
        if abs(drift_seconds) > tolerance_seconds:
            raise HostedDroidError(
                "isaac.step_simulation advanced the policy action by the wrong "
                f"duration: expected={expected_seconds:.9f}s, "
                f"observed={observed_seconds:.9f}s"
            )
        return (
            joint_positions,
            applied_joint_indices,
            {
                "before": before,
                "after": after,
                "stepped": stepped,
                "expected_simulation_seconds": expected_seconds,
                "observed_simulation_seconds": observed_seconds,
                "timeline_drift_seconds": drift_seconds,
                "joint_target_control_source": control_source,
                "contact_integrity": step.get("contact_integrity"),
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
        for attempt in range(_TRANSIENT_MCP_RETRIES):
            try:
                return self._call_once(mcp, name, arguments)
            except HostedDroidError as exc:
                if not _is_retryable_mcp_failure(name, exc):
                    raise
                if attempt + 1 == _TRANSIENT_MCP_RETRIES:
                    raise
                self._sleep(min(self.config.readiness_poll_seconds, 5.0))
        raise AssertionError("unreachable")

    def _call_once(
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


def _parse_task_state(
    payload: Any,
    spec: DroidTaskSuccessSpec,
) -> _DroidTaskState:
    if not isinstance(payload, Mapping):
        raise HostedDroidError("Isaac task state must be a mapping")
    if payload.get("object_path") != spec.object_prim_path:
        raise HostedDroidError("Isaac task state returned the wrong object prim")
    if payload.get("receptacle_path") != spec.receptacle_prim_path:
        raise HostedDroidError("Isaac task state returned the wrong receptacle prim")
    if payload.get("gripper_reference_path") != spec.gripper_reference_prim_path:
        raise HostedDroidError(
            "Isaac task state returned the wrong gripper reference prim"
        )
    velocity = payload.get("object_velocity")
    if not isinstance(velocity, Mapping):
        raise HostedDroidError("Isaac task state is missing object velocity")
    receptacle_velocity = payload.get("receptacle_velocity")
    if not isinstance(receptacle_velocity, Mapping):
        raise HostedDroidError("Isaac task state is missing receptacle velocity")
    velocity_source = payload.get("velocity_source")
    if velocity_source not in {"physics_tensor", "legacy_dynamic_control"}:
        raise HostedDroidError(
            "Isaac task state returned an unsupported velocity source"
        )
    return _DroidTaskState(
        object_bounds=_parse_axis_aligned_bounds(
            payload.get("object_bounds"),
            "object_bounds",
        ),
        receptacle_bounds=_parse_axis_aligned_bounds(
            payload.get("receptacle_bounds"),
            "receptacle_bounds",
        ),
        velocity_source=velocity_source,
        object_linear_velocity=_finite_triplet(
            velocity.get("linear"),
            "object_velocity.linear",
        ),
        object_angular_velocity=_finite_triplet(
            velocity.get("angular"),
            "object_velocity.angular",
        ),
        receptacle_linear_velocity=_finite_triplet(
            receptacle_velocity.get("linear"),
            "receptacle_velocity.linear",
        ),
        receptacle_angular_velocity=_finite_triplet(
            receptacle_velocity.get("angular"),
            "receptacle_velocity.angular",
        ),
        object_runtime_position=_finite_triplet(
            payload.get("object_runtime_position"),
            "object_runtime_position",
        ),
        gripper_reference_position=_finite_triplet(
            payload.get("gripper_reference_position"),
            "gripper_reference_position",
        ),
    )


def _parse_axis_aligned_bounds(value: Any, name: str) -> _AxisAlignedBounds:
    if not isinstance(value, Mapping):
        raise HostedDroidError(f"Isaac task state is missing {name}")
    minimum = _finite_triplet(value.get("minimum"), f"{name}.minimum")
    maximum = _finite_triplet(value.get("maximum"), f"{name}.maximum")
    if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
        raise HostedDroidError(f"Isaac task state returned inverted {name}")
    return _AxisAlignedBounds(minimum=minimum, maximum=maximum)


def _finite_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise HostedDroidError(f"Isaac task state {name} must be a length-3 list")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise HostedDroidError(f"Isaac task state {name} must contain finite numbers")
    return cast(tuple[float, float, float], tuple(float(item) for item in value))


def _raise_tool_error(name: str, payload: Mapping[str, Any]) -> None:
    status = str(payload.get("status", "")).lower()
    if payload.get("success") is False or status in {"error", "failed", "failure"}:
        message = payload.get("message") or payload.get("error") or "unknown error"
        raise HostedDroidError(f"{name} failed: {message}")


def _is_retryable_mcp_failure(name: str, exc: HostedDroidError) -> bool:
    if name == "isaac.step_simulation":
        return False
    message = str(exc)
    if any(marker in message for marker in _TRANSIENT_MCP_FAILURE_MARKERS):
        return True
    return name in _IDEMPOTENT_TRANSPORT_RETRY_TOOLS and any(
        marker in message for marker in _TRANSIENT_TRANSPORT_FAILURE_MARKERS
    )


def _is_legacy_camera_contract_failure(exc: HostedDroidError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _LEGACY_CAMERA_CONTRACT_MARKERS)


def _is_transport_failure(exc: HostedDroidError) -> bool:
    return any(marker in str(exc) for marker in _TRANSIENT_TRANSPORT_FAILURE_MARKERS)


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


def _decode_valid_rgb(
    encoded: str,
    *,
    camera_prim_path: str,
    expected_width: int,
    expected_height: int,
) -> tuple[bytes, np.ndarray]:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (expected_width, expected_height):
                raise HostedDroidError(
                    f"camera {camera_prim_path} returned {image.size[0]}x{image.size[1]}, "
                    f"expected {expected_width}x{expected_height}"
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except HostedDroidError:
        raise
    except Exception as exc:
        raise HostedDroidError(
            f"invalid RGB artifact for camera {camera_prim_path}: {exc}"
        ) from exc

    luminance = rgb.astype(np.float32).mean(axis=2)
    p99 = float(np.percentile(luminance, 99))
    stddev = float(luminance.std())
    non_dark_fraction = float(np.count_nonzero(luminance > 8.0) / luminance.size)
    non_white_fraction = float(
        np.count_nonzero(luminance < _NON_WHITE_LUMINANCE_CUTOFF) / luminance.size
    )
    if (
        p99 < _MIN_LUMINANCE_P99
        or stddev < _MIN_LUMINANCE_STDDEV
        or non_dark_fraction < _MIN_NON_DARK_FRACTION
        or non_white_fraction < _MIN_NON_WHITE_FRACTION
    ):
        raise HostedDroidError(
            f"camera {camera_prim_path} returned an unrendered or low-information frame "
            f"(p99={p99:.2f}, stddev={stddev:.2f}, "
            f"non_dark_fraction={non_dark_fraction:.4f}, "
            f"non_white_fraction={non_white_fraction:.4f})"
        )
    return raw, np.ascontiguousarray(rgb)


def _response_field(response: Any, name: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def _tensor_array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif isinstance(value, Mapping) and "data" in value:
        shape = value.get("shape")
        value = np.asarray(value["data"])
        if shape is not None:
            value = value.reshape(shape)
    array = np.asarray(value)
    if array.dtype.kind not in {"b", "f", "i", "u"}:
        raise HostedDroidError(f"{name} must contain numeric tensor data")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise HostedDroidError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_policy_response(
    base_model: str,
    response: Any,
    action_chunk: np.ndarray,
) -> None:
    if base_model != "pi0-droid":
        return
    metadata = _response_field(response, "policy_metadata")
    mismatched_profile_keys = (
        sorted(_PI0_DROID_POLICY_PROFILE)
        if not isinstance(metadata, Mapping)
        else sorted(
            key
            for key, expected in _PI0_DROID_POLICY_PROFILE.items()
            if metadata.get(key) != expected
        )
    )
    if mismatched_profile_keys:
        raise HostedDroidError(
            "pi0-droid response did not prove the pinned joint-position policy "
            f"profile; mismatched keys: {mismatched_profile_keys}"
        )
    if action_chunk.shape != (10, 8):
        raise HostedDroidError(
            f"pi0-droid must return action_chunk [10,8], got {list(action_chunk.shape)}"
        )


def _config_dict(config: HostedDroidConfig) -> dict[str, Any]:
    return {
        "environment_uri": config.environment_uri,
        "session_id": config.session_id,
        "base_model": config.base_model,
        "instruction": config.instruction,
        "robot_prim_path": config.robot_prim_path,
        "robot_usd_path": config.robot_usd_path,
        "cameras": [
            {
                "prim_path": camera.prim_path,
                "position": list(camera.position),
                "orientation_wxyz": list(camera.orientation_wxyz),
                "focal_length": camera.focal_length,
                "clipping_range": list(camera.clipping_range),
                "focus_distance": camera.focus_distance,
                "horizontal_aperture": camera.horizontal_aperture,
                "vertical_aperture": camera.vertical_aperture,
            }
            for camera in config.cameras
        ],
        "image_width": config.image_width,
        "image_height": config.image_height,
        "max_action_steps": config.max_action_steps,
        "open_loop_horizon": config.open_loop_horizon,
        "physics_steps_per_action": config.physics_steps_per_action,
        "target_control_hz": config.target_control_hz,
        "physics_hz": config.physics_hz,
        "solver_position_iterations": config.solver_position_iterations,
        "solver_velocity_iterations": config.solver_velocity_iterations,
        "runtime_provider": config.runtime_provider,
        "action_source": config.action_source,
        "replay_source_sha256": config.replay_source_sha256,
        "policy_mode": config.policy_mode,
        "include_predicted_video": config.include_predicted_video,
        "request_timeout_seconds": config.request_timeout_seconds,
        "launch_timeout_seconds": config.launch_timeout_seconds,
        "readiness_timeout_seconds": config.readiness_timeout_seconds,
        "readiness_poll_seconds": config.readiness_poll_seconds,
        "keep_session": config.keep_session,
        "record_video": config.record_video,
        "video_fps": config.video_fps,
        "results_dir": str(config.results_dir) if config.results_dir else None,
        "task_success": (
            {
                "name": config.task_success.name,
                "object_prim_path": config.task_success.object_prim_path,
                "receptacle_prim_path": config.task_success.receptacle_prim_path,
                "left_finger_prim_path": (config.task_success.left_finger_prim_path),
                "right_finger_prim_path": (config.task_success.right_finger_prim_path),
                "gripper_reference_prim_path": (
                    config.task_success.gripper_reference_prim_path
                ),
                "minimum_lift_meters": config.task_success.minimum_lift_meters,
                "minimum_lift_checks": config.task_success.minimum_lift_checks,
                "horizontal_containment_margin_meters": (
                    config.task_success.horizontal_containment_margin_meters
                ),
                "horizontal_containment_fraction": (
                    config.task_success.horizontal_containment_fraction
                ),
                "minimum_insertion_meters": (
                    config.task_success.minimum_insertion_meters
                ),
                "vertical_tolerance_meters": (
                    config.task_success.vertical_tolerance_meters
                ),
                "maximum_linear_speed_mps": (
                    config.task_success.maximum_linear_speed_mps
                ),
                "maximum_angular_speed_rps": (
                    config.task_success.maximum_angular_speed_rps
                ),
                "maximum_object_displacement_per_action_meters": (
                    config.task_success.maximum_object_displacement_per_action_meters
                ),
                "maximum_receptacle_size_change_meters": (
                    config.task_success.maximum_receptacle_size_change_meters
                ),
                "gripper_closed_threshold": (
                    config.task_success.gripper_closed_threshold
                ),
                "gripper_released_threshold": (
                    config.task_success.gripper_released_threshold
                ),
                "maximum_contact_penetration_meters": (
                    config.task_success.maximum_contact_penetration_meters
                ),
                "maximum_contact_normal_impulse_ns": (
                    config.task_success.maximum_contact_normal_impulse_ns
                ),
                "maximum_closed_support_loss_meters": (
                    config.task_success.maximum_closed_support_loss_meters
                ),
                "minimum_bilateral_contact_fraction": (
                    config.task_success.minimum_bilateral_contact_fraction
                ),
                "maximum_bilateral_contact_gap_updates": (
                    config.task_success.maximum_bilateral_contact_gap_updates
                ),
                "maximum_bilateral_normal_dot": (
                    config.task_success.maximum_bilateral_normal_dot
                ),
                "required_settled_checks": (
                    config.task_success.required_settled_checks
                ),
            }
            if config.task_success is not None
            else None
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _evidence_frame_name(sample_index: int, camera_index: int) -> str:
    return f"sample-{sample_index:05d}-{_EVIDENCE_CAMERA_NAMES[camera_index]}.png"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
