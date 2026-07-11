"""Import-light DROID observation extraction and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _camera(value: Any, name: str) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an RGB image [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        if image.dtype.kind not in {"i", "u"} or image.size == 0:
            raise ValueError(f"{name} must contain uint8-compatible RGB values")
        if image.min() < 0 or image.max() > 255:
            raise ValueError(f"{name} RGB values must be in [0, 255]")
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector)


@dataclass(frozen=True)
class DroidObservation:
    """One raw DROID policy observation with the simulator batch removed."""

    exterior_image_1_left: np.ndarray
    exterior_image_2_left: np.ndarray
    wrist_image_left: np.ndarray
    joint_position: np.ndarray
    gripper_position: np.ndarray
    instruction: str

    @classmethod
    def from_sim_observation(
        cls, observation: Mapping[str, Any], instruction: str
    ) -> "DroidObservation":
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        policy = observation.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("observation must contain a 'policy' mapping")
        return cls(
            exterior_image_1_left=_camera(policy["external_cam"], "external_cam"),
            exterior_image_2_left=_camera(policy["external_cam_2"], "external_cam_2"),
            wrist_image_left=_camera(policy["wrist_cam"], "wrist_cam"),
            joint_position=_vector(policy["arm_joint_pos"], 7, "arm_joint_pos"),
            gripper_position=_vector(policy["gripper_pos"], 1, "gripper_pos"),
            instruction=instruction,
        )


def camera_strip(observation: DroidObservation, size: int = 224) -> np.ndarray:
    """Build a compact three-camera visualization without OpenCV or Isaac Sim."""

    def resize_nearest(image: np.ndarray) -> np.ndarray:
        rows = np.linspace(0, image.shape[0] - 1, size, dtype=np.int64)
        columns = np.linspace(0, image.shape[1] - 1, size, dtype=np.int64)
        return image[rows[:, None], columns]

    return np.concatenate(
        [
            resize_nearest(observation.exterior_image_1_left),
            resize_nearest(observation.exterior_image_2_left),
            resize_nearest(observation.wrist_image_left),
        ],
        axis=1,
    )
