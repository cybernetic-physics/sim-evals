import tyro
import numpy as np
from PIL import Image
# from openpi_client import websocket_client_policy, image_tools
from openpi_client import image_tools
from .test import WebsocketClientPolicy

from .abstract_client import InferenceClient


class Client(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        num_envs: int = 1,
    ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.num_envs = num_envs
        # self.client = websocket_client_policy.WebsocketClientPolicy(
        self.client = WebsocketClientPolicy(
            remote_host, remote_port
        )

        # Per-env chunk state
        self._chunk_counters = np.zeros(num_envs, dtype=np.int32)
        self._action_chunks = None  # (num_envs, chunk_len, action_dim)

    def reset(self, env_ids: list[int] | None = None):
        """Reset chunk state. Pass env_ids to reset specific envs, or None for all."""
        if env_ids is None:
            self._chunk_counters[:] = 0
            self._action_chunks = None
        else:
            # Force re-query on next infer for these envs
            self._chunk_counters[env_ids] = self.open_loop_horizon

    def infer(self, obs: dict, instruction: list[str]) -> dict:
        """Batched inference. Returns actions (num_envs, action_dim) and viz (num_envs, H, W*2, 3)."""
        assert isinstance(instruction, list), "Instruction must be a list of strings"
        curr_obs = self._extract_observation(obs)  # all arrays are (num_envs, ...)

        needs_new = self._chunk_counters >= self.open_loop_horizon
        if self._action_chunks is None or needs_new.any():
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            new_chunks = self.client.infer(request_data)["actions"]  # (num_envs, chunk_len, action_dim)
            if self._action_chunks is None:
                self._action_chunks = new_chunks.copy()
                self._chunk_counters[:] = 0
            else:
                # Only update envs that needed new chunks
                update_ids = np.where(needs_new)[0]
                self._action_chunks[update_ids] = new_chunks[update_ids]
                self._chunk_counters[update_ids] = 0

        # Gather current action per env
        actions = self._action_chunks[np.arange(self.num_envs), self._chunk_counters]  # (num_envs, action_dim)
        self._chunk_counters += 1

        # Binarize gripper action
        gripper = (actions[:, -1] > 0.5).astype(np.float32)
        actions = np.concatenate([actions[:, :-1], gripper[:, None]], axis=1)

        # Viz per env
        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)  # (num_envs, H, W, 3)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        viz = np.concatenate([img1, img2], axis=2)  # (num_envs, H, W*2, 3)

        return {"action": actions, "viz": viz}

    def _extract_observation(self, obs_dict):
        """Extract observations, preserving batch dim (num_envs, ...)."""
        right_image = obs_dict["policy"]["external_cam"].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"].clone().detach().cpu().numpy()
        joint_position = obs_dict["policy"]["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = obs_dict["policy"]["gripper_pos"].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

if __name__ == "__main__":
    import torch
    args = tyro.cli(Args)
    client = Client(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
