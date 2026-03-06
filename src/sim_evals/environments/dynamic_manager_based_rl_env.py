from pathlib import Path
import json
import torch
from typing import Any
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class DynamicManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        *args,
        usd_file: str | None = None,
        progress_criteria: list[tuple[Any, list[int]]] | None = None,
        **kwargs,
    ):
        self.usd_file = usd_file
        self.instruction = ""
        cfg.dynamic_setup(usd_file=usd_file, progress_criteria=progress_criteria)

        if usd_file is not None and (Path(usd_file).parent / "initial_conditions.json").exists():
            with open(Path(usd_file).parent / "initial_conditions.json", "r") as f:
                self.instruction = json.load(f)["instruction"]

        super().__init__(cfg=cfg, *args, **kwargs)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        info["instruction"] = self.instruction
        return obs, info
