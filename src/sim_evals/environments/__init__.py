import gymnasium as gym
from .droid_environment import EnvCfg as DroidEnvCfg
from .lw_environment import EnvCfg as LWEnvCfg
from .dynamic_manager_based_rl_env import DynamicManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnv
import os
from pathlib import Path
from . import mdp


DATA_PATH = Path(os.environ.get("POLARIS_DATA_PATH", Path(__file__).parent.parent.parent.parent / "assets"))

gym.register(
    id="DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEnvCfg,
    },
    disable_env_checker=True,
)


# TODO: check that final points are awarded when object is not grasped
gym.register(
    id="Test",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "scene1.usd"),
    },
    disable_env_checker=True,
)

gym.register(
    id="BlockStack",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_block_stack/scene.usda"),
        "progress_criteria": [
            mdp.reach("green_cube", threshold=0.2),
            mdp.reach("wood_cube", threshold=0.2),
            (mdp.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
            (mdp.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
            (mdp.point_in_obb("green_cube", "tray", check_axes=(0, 1)), [2]),
            (mdp.point_in_obb("wood_cube", "tray", check_axes=(0, 1)), [3]),
            (mdp.point_in_obb("green_cube", "wood_cube", check_axes=(0, 1)), [4, 5]),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="FoodBussing",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_food_bus/scene.usda"),
        "progress_criteria": [
            mdp.reach("ice_cream_", threshold=0.2),
            mdp.reach("grapes", threshold=0.2),
            (mdp.lift("ice_cream_", threshold=0.06), [0]),
            (mdp.lift("grapes", threshold=0.06), [1]),
            (mdp.point_in_obb("ice_cream_", "bowl", check_axes=(0, 1, 2)), [2]),
            (mdp.point_in_obb("grapes", "bowl", check_axes=(0, 1, 2)), [3]),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="PanClean",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_pan_clean/scene.usda"),
        "progress_criteria": [
            mdp.reach("sponge", threshold=0.2),
            (mdp.lift("sponge", threshold=0.09, default_height=0.0), [0]),
            (mdp.point_in_obb("sponge", "pan", check_axes=(0, 1)), [1]),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="OrganizeTools",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "OrganizeTools_Sence/scene.usda"),
    },
    disable_env_checker=True,
)

gym.register(
    id="MoveLatteCup",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "RA_LW_Scene/MoveLatteCup_Sence/scene.usda"),
    },
    disable_env_checker=True,
)

gym.register(
    id="TapeIntoContainer",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "TapeIntoContainer_Scene/scene.usda"),
    },
    disable_env_checker=True,
)