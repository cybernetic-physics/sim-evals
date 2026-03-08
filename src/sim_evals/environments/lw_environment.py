import torch
import isaaclab.sim as sim_utils
from typing import Any
# import isaaclab.envs.mdp as mdp
from . import mdp
import numpy as np
import json
from functools import partial

from typing import List
from pathlib import Path
from pxr import Usd, UsdPhysics, Gf

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm, ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG

from .nvidia_droid import NVIDIA_DROID

DATA_PATH = Path(__file__).parent / "../../../assets/"

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    robot = NVIDIA_DROID

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.4, 0.55), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

    external_cam_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, -0.4, 0.55), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def __post_init__(
        self,
    ):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
    def dynamic_setup(self, usd_file: str, **kwargs):
        environment_path_ = Path(usd_file)
        environment_path = str(environment_path_.resolve())
        print(f"Setting up scene from {environment_path}")

        self.scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=environment_path,
                activate_contact_sensors=False,
            ),
        )
        stage = Usd.Stage.Open(environment_path)
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            name = child.GetName()

            if UsdPhysics.RigidBodyAPI(child) or any([UsdPhysics.RigidBodyAPI(c) for c in child.GetChildren()]):
                print(f"Rigid Body: {name}")
                pos = child.GetAttribute("xformOp:translate").Get()
                # Try orient (quaternion) first, fallback to rotateXYZ (Euler degrees)
                orient_val = child.GetAttribute("xformOp:orient").Get()
                if orient_val is not None:
                    rot = (
                        orient_val.GetReal(),
                        orient_val.GetImaginary()[0],
                        orient_val.GetImaginary()[1],
                        orient_val.GetImaginary()[2],
                    )
                else:
                    rotate_val = child.GetAttribute("xformOp:rotateXYZ").Get()
                    if rotate_val is not None:
                        # Euler XYZ in degrees -> quaternion via composed Gf.Rotations
                        rx = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotate_val[0])
                        ry = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotate_val[1])
                        rz = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotate_val[2])
                        combined = rz * ry * rx
                        q = combined.GetQuat()
                        rot = (q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2])
                    else:
                        rot = (1.0, 0.0, 0.0, 0.0)
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=pos,
                        rot=rot,
                    ),
                )
                setattr(self, name, asset)

    

class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction

@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        external_cam_2 = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_2"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


class reset_initial_conditions(ManagerTermBase):
    def __init__(self, cfg: EventTerm, env: ManagerBasedRLEnv):
        initial_conditions_file = cfg.params.get("initial_conditions_file", [])

        with open(initial_conditions_file, "r") as f:
            self.initial_conditions = json.load(f)["poses"]

        self.current_index = 0
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv, env_ids, initial_conditions_file: str):
        if len(self.initial_conditions) == 0:
            print(f"No initial conditions found. Objects will be reset to default poses.")
            return

        # Ensure env_ids is a tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=env.device)
        num_resets = len(env_ids)

        # Get ICs for each env being reset
        ic_indices = [(self.current_index + i) % len(self.initial_conditions) for i in range(num_resets)]
        print(f"Resetting envs {env_ids.cpu().tolist()} to initial conditions {ic_indices}")

        # Get env origins for all envs being reset
        env_origins = env.scene.env_origins[env_ids]  # (num_resets, 3)

        # Collect all object names (assume all ICs have same objects)
        obj_names = list(self.initial_conditions[0].keys())

        # Batch write per object
        for obj in obj_names:
            poses = []
            for i in range(num_resets):
                pose = self.initial_conditions[ic_indices[i]][obj]
                poses.append(pose)
            poses = torch.tensor(poses, device=env.device)  # (num_resets, 7)
            # Add env origin offset to positions
            poses[:, :3] += env_origins
            env.scene[obj].write_root_pose_to_sim(poses, env_ids=env_ids)

        # Advance counter by number of envs reset
        self.current_index += num_resets

@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    def dynamic_setup(self, usd_file: str, **kwargs):
        if (Path(usd_file).parent / "initial_conditions.json").exists():
            self.reset_initial_conditions = EventTerm(
                func=reset_initial_conditions, 
                mode="reset",
                params={
                    "initial_conditions_file": Path(usd_file).parent / "initial_conditions.json",
                }
            )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    def dynamic_setup(self, progress_criteria: list[tuple[Any, list[int]]], **kwargs):
        if progress_criteria is not None:
            self.progress = RewTerm(
                func=mdp.rubric_reward,
                weight=1.0,
                params=dict(
                    criteria=progress_criteria,
                )
            )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.wait_for_textures = True
        self.rerender_on_reset = True

        # overwrite carb settings
        carb_settings = self.sim.render.carb_settings if self.sim.render.carb_settings is not None else {}
        carb_settings["rtx.post.tonemap.op"] = "Iray"
        carb_settings["rtx/post/tonemap/irayReinhard/crushBlacks"] = 0.2
        carb_settings["rtx/post/tonemap/irayReinhard/burnHighlights"] = 0.1
        carb_settings["rtx/post/tonemap/enableSrgbToGamma"] = False
        carb_settings["rtx/post/tonemap/cm2Factor"] = 1.2
        self.sim.render.carb_settings = carb_settings

    
    def dynamic_setup(self, **kwargs):
        self.scene.dynamic_setup(**kwargs)
        self.events.dynamic_setup(**kwargs)
        self.rewards.dynamic_setup(**kwargs)


