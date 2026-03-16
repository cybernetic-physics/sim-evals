"""Reward term functions for manager-based RL environments."""

from __future__ import annotations

import torch
import numpy as np
from typing import Callable
import isaacsim.core.utils.bounds as bounds_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.assets import RigidObject
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reach(obj_name, threshold=0.05):
    """Returns a checker: (env) -> (num_envs,) bool tensor."""

    def checker(env):
        obj_pos = env.scene[obj_name].data.root_pos_w  # (num_envs, 3)
        ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # (num_envs, 3)
        dist = torch.norm(obj_pos - ee_pos, dim=-1)  # (num_envs,)
        return dist < threshold

    return checker


def proximity(obj_name, target_name, threshold=0.1, check_axes=(0, 1), require_gripper_open=False, gripper_open_threshold=0.05):
    """Returns a checker: (env) -> (num_envs,) bool tensor.

    Tests if obj's centroid is within threshold distance of target's centroid
    along the specified axes.

    Args:
        obj_name: Scene entity name of the object to test.
        target_name: Scene entity name of the target object.
        threshold: Maximum distance along checked axes.
        check_axes: Which world axes to check distance on. Default (0, 1) = XY.
        require_gripper_open: If True, only counts when gripper is open.
        gripper_open_threshold: Gripper joint pos (normalized 0-1) below which
            the gripper is considered open.
    """

    def checker(env):
        obj_pos = env.scene[obj_name].data.root_pos_w  # (num_envs, 3)
        target_pos = env.scene[target_name].data.root_pos_w  # (num_envs, 3)
        diff = obj_pos - target_pos
        axes = list(check_axes)
        dist = torch.norm(diff[:, axes], dim=-1)
        result = dist < threshold

        if require_gripper_open:
            robot = env.scene["robot"]
            finger_idx = robot.data.joint_names.index("finger_joint")
            gripper_pos = robot.data.joint_pos[:, finger_idx] / (np.pi / 4)
            result = result & (gripper_pos < gripper_open_threshold)

        return result

    return checker


def lift(obj_name, threshold=0.05, default_height=None):
    """Returns a checker: (env) -> (num_envs,) bool tensor."""
    _default_height = default_height

    def checker(env):
        nonlocal _default_height
        obj_pos = env.scene[obj_name].data.root_pos_w  # (num_envs, 3)
        if _default_height is None:
            _default_height = env.scene[obj_name].data.default_root_state[:, 2]  # (num_envs,)
        return (obj_pos[:, 2] - _default_height) > threshold  # (num_envs,)

    return checker


def point_in_obb(obj_name, receptacle_name, check_axes=(0, 1, 2), require_gripper_open=False, gripper_open_threshold=0.05):
    """Returns a checker: (env) -> (num_envs,) bool tensor.

    Tests if obj's centroid is inside receptacle's OBB.
    OBB is auto-computed from the USD mesh on first call.

    Args:
        obj_name: Scene entity name of the object whose centroid is tested.
        receptacle_name: Scene entity name of the receptacle whose OBB defines the region.
        check_axes: OBB axis indices to check. Default (0, 1, 2) = all three.
        require_gripper_open: If True, only counts when gripper is open.
        gripper_open_threshold: Gripper joint pos (normalized 0-1) below which
            the gripper is considered open. Default 0.1.
    """
    _obb_cache = {}

    def checker(env):
        if not _obb_cache:
            receptacle = env.scene[receptacle_name]
            bbox_cache = bounds_utils.create_bbox_cache()
            c, a, h = _compute_obb_body_frame(receptacle, bbox_cache)
            _obb_cache["centroid_body"] = c
            _obb_cache["axes_body"] = a
            _obb_cache["half_extents"] = h

        receptacle = env.scene[receptacle_name]
        centroids_w, axes_w = _obb_to_world(
            _obb_cache["centroid_body"],
            _obb_cache["axes_body"],
            receptacle.data.root_pos_w,
            receptacle.data.root_quat_w,
            env.num_envs,
        )
        obj_pos = env.scene[obj_name].data.root_pos_w
        inside = _check_point_in_obb(
            obj_pos,
            centroids_w,
            axes_w,
            _obb_cache["half_extents"],
            check_axes=check_axes,
        )

        # Debug
        if env.episode_length_buf[0] % 100 == 0:
            d = obj_pos - centroids_w
            proj = torch.abs(torch.bmm(d.unsqueeze(1), axes_w.transpose(1, 2)).squeeze(1))
            he = _obb_cache["half_extents"]
            print(f"[point_in_obb] {obj_name}->{receptacle_name}: "
                  f"obj={obj_pos[0].tolist()}, center={centroids_w[0].tolist()}, "
                  f"proj={proj[0].tolist()}, he={he.tolist()}, inside={inside[0].item()}", end="")

        if require_gripper_open:
            robot = env.scene["robot"]
            finger_idx = robot.data.joint_names.index("finger_joint")
            gripper_pos = robot.data.joint_pos[:, finger_idx] / (np.pi / 4)  # normalize to 0-1
            gripper_open = gripper_pos < gripper_open_threshold
            if env.episode_length_buf[0] % 100 == 0:
                print(f", gripper={gripper_pos[0].item():.3f}, open={gripper_open[0].item()}", end="")
            inside = inside & gripper_open

        if env.episode_length_buf[0] % 100 == 0:
            print(f", final={inside[0].item()}")

        return inside

    return checker


def _check_point_in_obb(
    points: torch.Tensor,
    centroids: torch.Tensor,
    axes: torch.Tensor,
    half_extents: torch.Tensor,
    check_axes: tuple[int, ...] = (0, 1, 2),
) -> torch.Tensor:
    """Check if points are inside Oriented Bounding Boxes along specified axes.

    Args:
        points: Points to test (num_envs, 3).
        centroids: OBB centers (num_envs, 3).
        axes: OBB orientation axes (num_envs, 3, 3).
        half_extents: OBB half-extents (3,).
        check_axes: Which OBB axis indices to check. Default (0, 1, 2) = all three.

    Returns:
        Boolean tensor (num_envs,).
    """
    d = points - centroids  # (num_envs, 3)
    projections = torch.abs(torch.bmm(d.unsqueeze(1), axes.transpose(1, 2)).squeeze(1))  # (num_envs, 3)
    check = list(check_axes)
    return (projections[:, check] <= half_extents[check].unsqueeze(0)).all(dim=1)


def _compute_obb_body_frame(
    obj: RigidObject,
    bbox_cache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an object's OBB and convert it to body frame (cached once at init).

    Returns:
        (centroid_body, axes_body, half_extents) all as tensors on the object's device.
    """
    base_path = obj.cfg.prim_path.replace(".*", "0", 1)
    centroid_world, axes_world, half_extents = bounds_utils.compute_obb(bbox_cache, base_path)

    device = obj.device
    # Use default root state (matches USD stage) rather than current root_pos_w
    # (which may have been moved by initial conditions that the USD stage doesn't reflect)
    pos_world = obj.data.default_root_state[0, :3]
    quat_world = obj.data.default_root_state[0, 3:7]

    centroid_world_t = torch.tensor(centroid_world, device=device, dtype=torch.float32)
    axes_world_t = torch.tensor(axes_world, device=device, dtype=torch.float32)

    # Transform centroid into body frame
    centroid_body = quat_apply_inverse(quat_world, centroid_world_t - pos_world)

    # Transform axes into body frame
    rot_matrix_world = matrix_from_quat(quat_world.unsqueeze(0))[0]  # (3, 3)
    axes_body = torch.matmul(rot_matrix_world.T, axes_world_t.T).T  # (3, 3)

    half_extents_t = torch.tensor(half_extents, device=device, dtype=torch.float32)

    return centroid_body, axes_body, half_extents_t


def _obb_to_world(
    centroid_body: torch.Tensor,
    axes_body: torch.Tensor,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform cached body-frame OBB to world frame for all envs.

    Returns:
        (centroids_world, axes_world) with shapes (num_envs, 3) and (num_envs, 3, 3).
    """
    centroids_world = pos_w + quat_apply(
        quat_w, centroid_body.unsqueeze(0).expand(num_envs, -1)
    )
    rot_matrices = matrix_from_quat(quat_w)  # (num_envs, 3, 3)
    axes_world = torch.bmm(
        rot_matrices,
        axes_body.unsqueeze(0).expand(num_envs, -1, -1).transpose(1, 2),
    ).transpose(1, 2)  # (num_envs, 3, 3)
    return centroids_world, axes_world


# # ---------------------------------------------------------------------------
# # Reward terms (ManagerTermBase classes)
# # ---------------------------------------------------------------------------

class PointInOBBReward(ManagerTermBase):
    """1.0 when object A's centroid is inside object B's OBB, 0.0 otherwise.

    OBB is auto-computed from the USD mesh at init — no manual extents needed.
    Supports checking a subset of axes via ``check_axes`` (e.g. ``(0, 1)`` for
    the xy-plane of the OBB, ignoring vertical).

    Params:
        asset_cfg_a: The "insertive" object whose centroid is tested.
        asset_cfg_b: The "receptacle" object whose OBB defines the region.
        check_axes: Tuple of OBB axis indices to check. Default ``(0, 1, 2)``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset_a = env.scene[cfg.params["asset_cfg_a"].name]
        self._asset_b = env.scene[cfg.params["asset_cfg_b"].name]

        bbox_cache = bounds_utils.create_bbox_cache()
        self._centroid_body, self._axes_body, self._half_extents = _compute_obb_body_frame(
            self._asset_b, bbox_cache
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_a: SceneEntityCfg,
        asset_cfg_b: SceneEntityCfg,
        check_axes: tuple[int, ...] = (0, 1, 2),
    ) -> torch.Tensor:
        centroids_w, axes_w = _obb_to_world(
            self._centroid_body,
            self._axes_body,
            self._asset_b.data.root_pos_w,
            self._asset_b.data.root_quat_w,
            env.num_envs,
        )
        inside = _check_point_in_obb(
            self._asset_a.data.root_pos_w,
            centroids_w,
            axes_w,
            self._half_extents,
            check_axes=check_axes,
        )
        return inside.float()


class obb_overlap(ManagerTermBase):
    """1.0 when the OBBs of two objects overlap (all corners of A tested against B's OBB, and vice versa).

    Uses a conservative check: overlap is detected if ANY corner of either object
    is inside the other's OBB. This is not a full SAT test but is efficient and
    sufficient for typical manipulation reward signals.

    Params:
        asset_cfg_a: First object.
        asset_cfg_b: Second object.
        check_axes: Tuple of OBB axis indices to check. Default ``(0, 1, 2)``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset_a = env.scene[cfg.params["asset_cfg_a"].name]
        self._asset_b = env.scene[cfg.params["asset_cfg_b"].name]

        bbox_cache = bounds_utils.create_bbox_cache()

        self._centroid_body_a, self._axes_body_a, self._he_a = _compute_obb_body_frame(
            self._asset_a, bbox_cache
        )
        self._centroid_body_b, self._axes_body_b, self._he_b = _compute_obb_body_frame(
            self._asset_b, bbox_cache
        )

        # Precompute the 8 corner offsets in body frame for each object
        self._corners_body_a = self._corner_offsets(self._axes_body_a, self._he_a)  # (8, 3)
        self._corners_body_b = self._corner_offsets(self._axes_body_b, self._he_b)

    @staticmethod
    def _corner_offsets(axes: torch.Tensor, he: torch.Tensor) -> torch.Tensor:
        """Compute 8 corner offsets from centroid in body frame. Returns (8, 3)."""
        signs = torch.tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
             [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            device=axes.device, dtype=torch.float32,
        )  # (8, 3)
        # each corner = sum_i (sign_i * he_i * axis_i)
        scaled_axes = axes * he.unsqueeze(1)  # (3, 3) — row i = he_i * axis_i
        return signs @ scaled_axes  # (8, 3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_a: SceneEntityCfg,
        asset_cfg_b: SceneEntityCfg,
        check_axes: tuple[int, ...] = (0, 1, 2),
    ) -> torch.Tensor:
        n = env.num_envs

        # OBB B in world frame
        centroid_b_w, axes_b_w = _obb_to_world(
            self._centroid_body_b, self._axes_body_b,
            self._asset_b.data.root_pos_w, self._asset_b.data.root_quat_w, n,
        )

        # Corners of A in world frame
        centroid_a_w, _ = _obb_to_world(
            self._centroid_body_a, self._axes_body_a,
            self._asset_a.data.root_pos_w, self._asset_a.data.root_quat_w, n,
        )
        rot_a = matrix_from_quat(self._asset_a.data.root_quat_w)  # (n, 3, 3)
        corners_a_w = centroid_a_w.unsqueeze(1) + torch.bmm(
            self._corners_body_a.unsqueeze(0).expand(n, -1, -1), rot_a.transpose(1, 2)
        )  # (n, 8, 3)

        # Test each corner of A against OBB B
        any_inside = torch.zeros(n, device=env.device, dtype=torch.bool)
        for i in range(8):
            any_inside |= _check_point_in_obb(
                corners_a_w[:, i], centroid_b_w, axes_b_w, self._he_b, check_axes
            )

        # Symmetrically: corners of B against OBB A
        centroid_a_w_full, axes_a_w = _obb_to_world(
            self._centroid_body_a, self._axes_body_a,
            self._asset_a.data.root_pos_w, self._asset_a.data.root_quat_w, n,
        )
        rot_b = matrix_from_quat(self._asset_b.data.root_quat_w)
        corners_b_w = centroid_b_w.unsqueeze(1) + torch.bmm(
            self._corners_body_b.unsqueeze(0).expand(n, -1, -1), rot_b.transpose(1, 2)
        )

        for i in range(8):
            any_inside |= _check_point_in_obb(
                corners_b_w[:, i], centroid_a_w_full, axes_a_w, self._he_a, check_axes
            )

        return any_inside.float()


# ---------------------------------------------------------------------------
# Rubric-based progress reward
# ---------------------------------------------------------------------------

# A criterion is either:
#   - a callable(env) -> (num_envs,) bool tensor
#   - a tuple (callable, [dep_indices]) where deps must be reached first
Criterion = Callable[[ManagerBasedRLEnv], torch.Tensor] | tuple[Callable[[ManagerBasedRLEnv], torch.Tensor], list[int]]


class rubric_reward(ManagerTermBase):
    """Progress reward based on ordered criteria with optional dependencies.

    Each criterion is a callable ``(env) -> (num_envs,) bool`` or a
    ``(callable, [dep_indices])`` tuple where the criterion only counts once
    all its dependencies have been reached (ever, not necessarily right now).

    Returns ``num_reached_ever / num_criteria`` per env as the reward signal.
    State is automatically reset when an environment episode resets.

    Params:
        criteria: List of criteria (callables or (callable, dep_indices) tuples).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._criteria: list[Criterion] = cfg.params["criteria"]
        n_criteria = len(self._criteria)

        # (num_envs, num_criteria) — tracks whether each criterion was ever met
        self._reached = torch.zeros(env.num_envs, n_criteria, device=env.device, dtype=torch.bool)
        # Track previous episode length to detect resets
        self._prev_ep_len = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        criteria: list[Criterion],
    ) -> torch.Tensor:
        # Detect env resets: episode length decreased means that env was reset
        curr_ep_len = env.episode_length_buf
        reset_mask = curr_ep_len < self._prev_ep_len
        if reset_mask.any():
            self._reached[reset_mask] = False
        self._prev_ep_len[:] = curr_ep_len

        # Evaluate each criterion
        for idx, c in enumerate(self._criteria):
            if isinstance(c, tuple):
                fn, deps = c
                deps_met = self._reached[:, deps].all(dim=1)  # (num_envs,)
                result = fn(env) & deps_met
            else:
                result = c(env)

            # Sticky: once reached, stays reached until reset
            self._reached[:, idx] |= result

        # Progress = fraction of criteria ever reached
        # Divide by step_dt to cancel the reward manager's * dt scaling,
        # so the actual reward equals progress * weight.
        progress = self._reached.float().sum(dim=1) / len(self._criteria)
        return progress / env.step_dt
