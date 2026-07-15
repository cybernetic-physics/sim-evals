"""Worldline-local DSRL controller for a frozen hosted PI0-DROID policy.

This is a narrow PyTorch port of the public DSRL pixel-SAC controller. PI0
remains on the remote Worldlines sampler; only this small controller, critics,
replay, and optimizer state are mutable. The hosted method metadata records its
bounded replay and augmentation differences explicitly; the final VLM-token
feature is unavailable at the public black-box PI0 sampling boundary.
"""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import numpy.typing as npt
from PIL import Image

from sim_evals.inference.droid_observation import DroidObservation

if TYPE_CHECKING:
    from sim_evals.hosted_droid import DroidDsrlChunkTransition

DSRL_REFERENCE_REPOSITORY = "https://github.com/nakamotoo/dsrl_pi0"
DSRL_REFERENCE_COMMIT = "7f48937d4553e95244cd81c79236a3256df80597"
DSRL_METHOD_VARIANT = "dsrl_pixels_proprio_no_vlm_token_v1"
_PI0_BASE_POLICY_LINEAGE_KEYS = (
    "base_model",
    "openpi_config",
    "checkpoint_uri",
    "openpi_source_commit",
    "action_space",
    "action_horizon",
    "action_dim",
)
_TORCH_INITIALIZATION_LOCK = threading.RLock()


@dataclass(frozen=True)
class DsrlConfig:
    """Published DSRL defaults with bounded replay and startup controls."""

    image_size: int = 128
    action_dim: int = 32
    action_magnitude: float = 2.5
    encoder_features: tuple[int, ...] = (32, 32, 32, 32)
    encoder_strides: tuple[int, ...] = (3, 2, 2, 2)
    encoder_latent_dim: int = 50
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    num_critics: int = 2
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 1.0
    target_entropy: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 16
    replay_capacity: int = 2_048
    random_exploration_episodes: int = 1
    initial_updates: int = 5_000
    updates_per_transition: int = 30
    random_shift_pixels: int = 4
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_ints = {
            "image_size": self.image_size,
            "action_dim": self.action_dim,
            "encoder_latent_dim": self.encoder_latent_dim,
            "num_critics": self.num_critics,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "initial_updates": self.initial_updates,
            "updates_per_transition": self.updates_per_transition,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.action_dim != 32:
            raise ValueError("PI0-DROID DSRL action_dim must be 32")
        if not self.encoder_features:
            raise ValueError("encoder_features must not be empty")
        if len(self.encoder_features) != len(self.encoder_strides):
            raise ValueError("encoder feature and stride counts must match")
        if any(value < 1 for value in (*self.encoder_features, *self.encoder_strides)):
            raise ValueError("encoder features and strides must be positive")
        if not self.hidden_dims or any(value < 1 for value in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        for name, value in (
            ("action_magnitude", self.action_magnitude),
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("temperature_learning_rate", self.temperature_learning_rate),
            ("initial_temperature", self.initial_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.random_shift_pixels < 0:
            raise ValueError("random_shift_pixels must not be negative")
        if self.random_exploration_episodes < 0:
            raise ValueError("random_exploration_episodes must not be negative")
        if not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DsrlObservation:
    pixels: npt.NDArray[np.uint8]
    proprio: npt.NDArray[np.float32]

    @classmethod
    def from_droid(
        cls,
        observation: DroidObservation,
        *,
        image_size: int,
    ) -> "DsrlObservation":
        cameras = (
            observation.exterior_image_1_left,
            observation.exterior_image_2_left,
            observation.wrist_image_left,
        )
        resized = [_resize_rgb(image, image_size) for image in cameras]
        pixels = np.ascontiguousarray(np.concatenate(resized, axis=-1), dtype=np.uint8)
        joints = np.asarray(observation.joint_position, dtype=np.float32).reshape(7)
        gripper = np.asarray(observation.gripper_position, dtype=np.float32).reshape(1)
        # Match the public DSRL real-robot code's raw joint/gripper state. The
        # only deliberate feature omission in this first hosted variant is the
        # unavailable final PI0 VLM token.
        proprio = np.concatenate((joints, gripper))
        if not np.isfinite(proprio).all():
            raise ValueError("DSRL proprioception must contain only finite values")
        return cls(
            pixels=pixels, proprio=np.ascontiguousarray(proprio, dtype=np.float32)
        )


class DsrlReplayBuffer:
    """Fixed-capacity, controller-owned replay with compact uint8 pixels."""

    def __init__(self, config: DsrlConfig) -> None:
        self.capacity = config.replay_capacity
        image_shape = (config.image_size, config.image_size, 9)
        self.observations = np.empty((self.capacity, *image_shape), dtype=np.uint8)
        self.next_observations = np.empty((self.capacity, *image_shape), dtype=np.uint8)
        self.proprio = np.empty((self.capacity, 8), dtype=np.float32)
        self.next_proprio = np.empty((self.capacity, 8), dtype=np.float32)
        self.actions = np.empty((self.capacity, config.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.discounts = np.empty((self.capacity, 1), dtype=np.float32)
        self.masks = np.empty((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def insert(
        self,
        *,
        observation: DsrlObservation,
        action: npt.NDArray[np.float32],
        reward: float,
        discount: float,
        mask: float,
        next_observation: DsrlObservation,
    ) -> None:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.actions.shape[1:]:
            raise ValueError(
                f"DSRL replay action must have shape {self.actions.shape[1:]}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("DSRL replay action must contain only finite values")
        scalars = (reward, discount, mask)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("DSRL replay scalars must be finite")
        if not 0 <= discount <= 1:
            raise ValueError("DSRL replay discount must be in [0, 1]")
        if mask not in {0.0, 1.0}:
            raise ValueError("DSRL replay mask must be 0 or 1")
        index = self.cursor
        self.observations[index] = observation.pixels
        self.next_observations[index] = next_observation.pixels
        self.proprio[index] = observation.proprio
        self.next_proprio[index] = next_observation.proprio
        self.actions[index] = action_array
        self.rewards[index, 0] = reward
        self.discounts[index, 0] = discount
        self.masks[index, 0] = mask
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError("DSRL replay is empty")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "proprio": self.proprio[indices],
            "next_proprio": self.next_proprio[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "discounts": self.discounts[indices],
            "masks": self.masks[indices],
        }


class TorchDsrlController:
    """Small SAC controller whose lifecycle is independent of hosted PI0."""

    def __init__(self, config: DsrlConfig) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "Install sim-evals with the 'dsrl' extra to train a PI0 controller"
            ) from exc

        self.config = config
        self._torch = torch
        self._nn = nn
        self.device = _resolve_device(torch, config.device)
        self._rng = np.random.default_rng(config.seed)
        self._torch_generator = torch.Generator(device="cpu")
        self._torch_generator.manual_seed(config.seed)
        with _isolated_torch_initialization(torch, config.seed):
            self.actor = _build_actor(torch, nn, config).to(self.device)
            self.critic = _build_critic(torch, nn, config).to(self.device)
            _initialize_actor(nn, self.actor)
            _initialize_critic(nn, self.critic)
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(
                math.log(config.initial_temperature),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=config.temperature_learning_rate
        )
        self.replay = DsrlReplayBuffer(config)
        self.transitions = 0
        self.updates = 0
        self.training_trajectories = 0
        self.trained_transitions = 0
        self.base_policy_metadata: dict[str, Any] | None = None

    @property
    def gamma(self) -> float:
        return self.config.gamma

    def metadata(self) -> Mapping[str, Any]:
        trainable_parameters = (
            sum(
                parameter.numel()
                for module in (self.actor, self.critic)
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            + self.log_temperature.numel()
        )
        return {
            "method": DSRL_METHOD_VARIANT,
            "reference_repository": DSRL_REFERENCE_REPOSITORY,
            "reference_commit": DSRL_REFERENCE_COMMIT,
            "base_policy_frozen": True,
            "reference_deviations": [
                "final PI0 VLM token omitted because hosted sampling does not expose it",
                "replay is a bounded hosted ring instead of the reference dynamically growing buffer",
                "reference color jitter is omitted; random edge-padded shifts remain enabled",
            ],
            "device": str(self.device),
            "trainable_parameters": trainable_parameters,
            "transitions": self.transitions,
            "updates": self.updates,
            "training_trajectories": self.training_trajectories,
            "trained_transitions": self.trained_transitions,
            "replay_size": self.replay.size,
            "base_policy_metadata": self.base_policy_metadata,
            "config": self.config.to_dict(),
        }

    def select_action(
        self,
        observation: DroidObservation,
        *,
        deterministic: bool = False,
    ) -> npt.NDArray[np.float32]:
        encoded = DsrlObservation.from_droid(
            observation, image_size=self.config.image_size
        )
        if (
            not deterministic
            and self.training_trajectories < self.config.random_exploration_episodes
        ):
            action = self._torch.randn(
                (self.config.action_dim,),
                dtype=self._torch.float32,
                device="cpu",
                generator=self._torch_generator,
            ).numpy()
            return np.ascontiguousarray(action, dtype=np.float32)
        pixels, proprio = self._tensor_observation(encoded)
        with self._torch.no_grad():
            action, _ = self.actor.sample(
                pixels,
                proprio,
                deterministic=deterministic,
                generator=self._torch_generator,
            )
        result = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (32,) or not np.isfinite(result).all():
            raise RuntimeError("DSRL actor produced an invalid action")
        return np.ascontiguousarray(result)

    def record_transition(
        self,
        transition: "DroidDsrlChunkTransition",
    ) -> dict[str, Any]:
        """Insert one transition without changing the trajectory's collector."""
        base_policy_metadata = _base_policy_lineage(transition.policy_metadata)
        if self.base_policy_metadata is None:
            self.base_policy_metadata = base_policy_metadata
        elif self.base_policy_metadata != base_policy_metadata:
            raise ValueError("hosted PI0 base-policy lineage changed during DSRL run")
        observation = DsrlObservation.from_droid(
            transition.observation, image_size=self.config.image_size
        )
        next_observation = DsrlObservation.from_droid(
            transition.next_observation, image_size=self.config.image_size
        )
        self.replay.insert(
            observation=observation,
            action=np.asarray(transition.dsrl_action, dtype=np.float32),
            reward=transition.reward,
            discount=transition.discount,
            mask=transition.mask,
            next_observation=next_observation,
        )
        self.transitions += 1
        latest: dict[str, Any] = {
            "transitions": self.transitions,
            "updates": self.updates,
            "replay_size": self.replay.size,
        }
        return latest

    def train_after_trajectory(self, transition_count: int) -> dict[str, Any]:
        """Train only after every transition from one trajectory is in replay."""
        if isinstance(transition_count, bool) or transition_count < 1:
            raise ValueError("trajectory transition_count must be at least 1")
        untrained_transitions = self.transitions - self.trained_transitions
        if transition_count != untrained_transitions:
            raise ValueError(
                "trajectory transition_count does not match newly inserted replay"
            )
        update_steps = (
            self.config.initial_updates
            if self.updates == 0
            else transition_count * self.config.updates_per_transition
        )
        latest: dict[str, Any] = {
            "transitions": self.transitions,
            "updates": self.updates,
            "replay_size": self.replay.size,
        }
        for _ in range(update_steps):
            latest.update(self._update_once())
        self.trained_transitions = self.transitions
        self.training_trajectories += 1
        latest.update(
            {
                "training_trajectories": self.training_trajectories,
                "trained_transitions": self.trained_transitions,
                "trajectory_transitions": transition_count,
                "trajectory_updates": update_steps,
            }
        )
        return latest

    def save_checkpoint(
        self,
        path: Path,
        *,
        base_policy_metadata: Mapping[str, Any] | None = None,
        include_replay: bool = True,
    ) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        lineage = (
            _base_policy_lineage(base_policy_metadata)
            if base_policy_metadata is not None
            else self.base_policy_metadata
        )
        if lineage is None:
            raise ValueError("cannot checkpoint DSRL before PI0 lineage is observed")
        if (
            self.base_policy_metadata is not None
            and lineage != self.base_policy_metadata
        ):
            raise ValueError("requested checkpoint lineage does not match observed PI0")
        checkpoint_path = path / "controller.pt"
        temp_path = path / ".controller.pt.tmp"
        state = {
            "schema_version": 1,
            "method": DSRL_METHOD_VARIANT,
            "reference_repository": DSRL_REFERENCE_REPOSITORY,
            "reference_commit": DSRL_REFERENCE_COMMIT,
            "config": self.config.to_dict(),
            "base_policy_metadata": dict(lineage),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "transitions": self.transitions,
            "updates": self.updates,
            "training_trajectories": self.training_trajectories,
            "trained_transitions": self.trained_transitions,
            "numpy_rng_state": self._rng.bit_generator.state,
            "torch_generator_state": self._torch_generator.get_state(),
        }
        self._torch.save(state, temp_path)
        os.replace(temp_path, checkpoint_path)
        replay_path = None
        if include_replay:
            replay_path = path / "replay.npz"
            replay_temp = path / ".replay.npz.tmp"
            with replay_temp.open("wb") as output:
                np.savez_compressed(
                    output,
                    observations=self.replay.observations[: self.replay.size],
                    next_observations=self.replay.next_observations[: self.replay.size],
                    proprio=self.replay.proprio[: self.replay.size],
                    next_proprio=self.replay.next_proprio[: self.replay.size],
                    actions=self.replay.actions[: self.replay.size],
                    rewards=self.replay.rewards[: self.replay.size],
                    discounts=self.replay.discounts[: self.replay.size],
                    masks=self.replay.masks[: self.replay.size],
                    cursor=np.asarray([self.replay.cursor], dtype=np.int64),
                    size=np.asarray([self.replay.size], dtype=np.int64),
                )
            os.replace(replay_temp, replay_path)
        else:
            stale_replay = path / "replay.npz"
            if stale_replay.exists():
                stale_replay.unlink()
        manifest = {
            "schema_version": 1,
            "method": DSRL_METHOD_VARIANT,
            "reference_repository": DSRL_REFERENCE_REPOSITORY,
            "reference_commit": DSRL_REFERENCE_COMMIT,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "replay": replay_path.name if replay_path is not None else None,
            "replay_sha256": (
                _sha256_file(replay_path) if replay_path is not None else None
            ),
            "transitions": self.transitions,
            "updates": self.updates,
            "training_trajectories": self.training_trajectories,
            "trained_transitions": self.trained_transitions,
            "replay_size": self.replay.size if replay_path is not None else 0,
            "base_policy_metadata": dict(lineage),
            "config": self.config.to_dict(),
        }
        _atomic_json(path / "manifest.json", manifest)
        return manifest

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        device: str | None = None,
        expected_base_policy_metadata: Mapping[str, Any] | None = None,
        require_replay: bool = False,
    ) -> "TorchDsrlController":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Install sim-evals with the 'dsrl' extra to load a controller"
            ) from exc
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("DSRL checkpoint manifest is missing or invalid") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
            raise ValueError("unsupported DSRL checkpoint manifest schema")
        if (
            manifest.get("method") != DSRL_METHOD_VARIANT
            or manifest.get("reference_commit") != DSRL_REFERENCE_COMMIT
        ):
            raise ValueError("DSRL checkpoint manifest method does not match")
        if manifest.get("checkpoint") != "controller.pt":
            raise ValueError("DSRL checkpoint manifest has an invalid controller file")
        checkpoint_path = path / "controller.pt"
        expected_checkpoint_hash = manifest.get("checkpoint_sha256")
        if (
            not isinstance(expected_checkpoint_hash, str)
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != expected_checkpoint_hash
        ):
            raise ValueError("DSRL controller checkpoint hash does not match manifest")
        replay_name = manifest.get("replay")
        if replay_name not in {None, "replay.npz"}:
            raise ValueError("DSRL checkpoint manifest has an invalid replay file")
        if require_replay and replay_name is None:
            raise ValueError(
                "DSRL training resume requires a replay-bearing checkpoint"
            )
        replay_path = path / "replay.npz" if replay_name is not None else None
        if replay_path is not None:
            expected_replay_hash = manifest.get("replay_sha256")
            if (
                not isinstance(expected_replay_hash, str)
                or not replay_path.is_file()
                or _sha256_file(replay_path) != expected_replay_hash
            ):
                raise ValueError("DSRL replay checkpoint hash does not match manifest")
        try:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("unsupported DSRL controller checkpoint schema")
        if state.get("method") != DSRL_METHOD_VARIANT:
            raise ValueError("DSRL controller checkpoint method does not match")
        base_policy_metadata = state.get("base_policy_metadata")
        if not isinstance(base_policy_metadata, Mapping):
            raise ValueError("DSRL checkpoint base-policy lineage is missing")
        if (
            expected_base_policy_metadata is not None
            and base_policy_metadata
            != _base_policy_lineage(expected_base_policy_metadata)
        ):
            raise ValueError("DSRL checkpoint base-policy lineage does not match")
        raw_config = state.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("DSRL checkpoint config is missing")
        if json.loads(json.dumps(dict(raw_config))) != manifest.get("config"):
            raise ValueError("DSRL checkpoint config does not match manifest")
        if base_policy_metadata != manifest.get("base_policy_metadata"):
            raise ValueError(
                "DSRL checkpoint base-policy lineage does not match manifest"
            )
        config_values = dict(raw_config)
        if device is not None:
            config_values["device"] = device
        controller = cls(DsrlConfig(**config_values))
        controller.actor.load_state_dict(state["actor"])
        controller.critic.load_state_dict(state["critic"])
        controller.target_critic.load_state_dict(state["target_critic"])
        controller.actor_optimizer.load_state_dict(state["actor_optimizer"])
        controller.critic_optimizer.load_state_dict(state["critic_optimizer"])
        controller.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        controller.log_temperature.data.copy_(
            state["log_temperature"].to(controller.device)
        )
        _optimizer_to(controller.actor_optimizer, controller.device)
        _optimizer_to(controller.critic_optimizer, controller.device)
        _optimizer_to(controller.temperature_optimizer, controller.device)
        controller.transitions = int(state["transitions"])
        controller.updates = int(state["updates"])
        controller.training_trajectories = int(state["training_trajectories"])
        controller.trained_transitions = int(state["trained_transitions"])
        if controller.transitions != manifest.get(
            "transitions"
        ) or controller.updates != manifest.get("updates"):
            raise ValueError("DSRL checkpoint counters do not match manifest")
        if (
            controller.training_trajectories != manifest.get("training_trajectories")
            or controller.trained_transitions != manifest.get("trained_transitions")
            or controller.training_trajectories < 0
            or controller.trained_transitions < 0
            or controller.trained_transitions > controller.transitions
        ):
            raise ValueError("DSRL checkpoint trajectory counters do not match")
        controller.base_policy_metadata = dict(base_policy_metadata)
        controller._rng.bit_generator.state = state["numpy_rng_state"]
        torch_generator_state = state.get("torch_generator_state")
        if torch_generator_state is None:
            raise ValueError("DSRL checkpoint controller RNG state is missing")
        controller._torch_generator.set_state(torch_generator_state)
        if replay_path is not None:
            with np.load(replay_path, allow_pickle=False) as replay:
                size = int(replay["size"][0])
                cursor = int(replay["cursor"][0])
                if size < 0 or size > controller.replay.capacity:
                    raise ValueError("saved DSRL replay exceeds configured capacity")
                if cursor < 0 or cursor >= controller.replay.capacity:
                    raise ValueError("saved DSRL replay cursor is out of range")
                if size < controller.replay.capacity and cursor != size:
                    raise ValueError("saved DSRL replay cursor does not match its size")
                for name in (
                    "observations",
                    "next_observations",
                    "proprio",
                    "next_proprio",
                    "actions",
                    "rewards",
                    "discounts",
                    "masks",
                ):
                    target = getattr(controller.replay, name)
                    source = replay[name]
                    if source.shape != target[:size].shape:
                        raise ValueError(
                            f"saved DSRL replay {name} shape does not match"
                        )
                    target[:size] = source
                controller.replay.size = size
                controller.replay.cursor = cursor
                if size != manifest.get("replay_size"):
                    raise ValueError("DSRL replay size does not match manifest")
        elif manifest.get("replay_size") != 0:
            raise ValueError("DSRL manifest declares replay data without a replay file")
        return controller

    def _update_once(self) -> dict[str, Any]:
        torch = self._torch
        batch = self.replay.sample(self.config.batch_size, self._rng)
        pixels = self._pixels_tensor(batch["observations"], augment=True)
        next_pixels = self._pixels_tensor(batch["next_observations"], augment=True)
        proprio = self._float_tensor(batch["proprio"])
        next_proprio = self._float_tensor(batch["next_proprio"])
        actions = self._float_tensor(batch["actions"])
        rewards = self._float_tensor(batch["rewards"])
        discounts = self._float_tensor(batch["discounts"])
        masks = self._float_tensor(batch["masks"])

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(
                next_pixels,
                next_proprio,
                generator=self._torch_generator,
            )
            target_q_values = self.target_critic(
                next_pixels, next_proprio, next_actions
            )
            target_q = torch.min(target_q_values, dim=0).values
            target = rewards + discounts * masks * target_q

        q_values = self.critic(pixels, proprio, actions)
        critic_loss = torch.mean((q_values - target.unsqueeze(0)) ** 2)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self._soft_update_target()

        self.critic.requires_grad_(False)
        sampled_actions, log_probability = self.actor.sample(
            pixels,
            proprio,
            generator=self._torch_generator,
        )
        policy_q = torch.min(
            self.critic(pixels, proprio, sampled_actions), dim=0
        ).values
        temperature = self.log_temperature.exp()
        actor_loss = torch.mean(temperature.detach() * log_probability - policy_q)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        entropy = -log_probability.detach()
        temperature_loss = torch.mean(
            self.log_temperature.exp() * (entropy - self.config.target_entropy)
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.updates += 1
        return {
            "transitions": self.transitions,
            "updates": self.updates,
            "replay_size": self.replay.size,
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature": float(self.log_temperature.exp().detach().cpu()),
            "entropy": float(entropy.mean().cpu()),
            "target_q": float(target.mean().cpu()),
            "next_log_probability": float(next_log_probability.mean().cpu()),
        }

    def _soft_update_target(self) -> None:
        with self._torch.no_grad():
            for target, source in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
                strict=True,
            ):
                target.mul_(1 - self.config.tau).add_(source, alpha=self.config.tau)

    def _tensor_observation(self, observation: DsrlObservation):
        pixels = self._pixels_tensor(observation.pixels[np.newaxis, ...])
        proprio = self._float_tensor(observation.proprio[np.newaxis, ...])
        return pixels, proprio

    def _pixels_tensor(self, pixels: np.ndarray, *, augment: bool = False):
        tensor = self._torch.as_tensor(
            pixels,
            dtype=self._torch.float32,
            device=self.device,
        ).permute(0, 3, 1, 2)
        tensor = tensor / 255.0
        if augment and self.config.random_shift_pixels:
            tensor = _random_shift(
                self._torch,
                tensor,
                padding=self.config.random_shift_pixels,
                generator=self._torch_generator,
            )
        return tensor

    def _float_tensor(self, value: np.ndarray):
        return self._torch.as_tensor(
            value,
            dtype=self._torch.float32,
            device=self.device,
        )


def _build_actor(torch: Any, nn: Any, config: DsrlConfig):
    encoder = _build_encoder(torch, nn, config)

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.trunk = _mlp(
                nn,
                config.encoder_latent_dim + 8,
                config.hidden_dims[:-1],
                config.hidden_dims[-1],
                activate_final=True,
            )
            self.mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
            self.log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)
            nn.init.orthogonal_(self.mean.weight, gain=1e-2)
            nn.init.zeros_(self.mean.bias)
            nn.init.orthogonal_(self.log_std.weight, gain=1e-2)
            nn.init.zeros_(self.log_std.bias)

        def distribution_parameters(self, pixels: Any, proprio: Any):
            encoded = self.encoder(pixels)
            hidden = self.trunk(torch.cat((encoded, proprio), dim=-1))
            return self.mean(hidden), torch.clamp(self.log_std(hidden), -20, 2)

        def sample(
            self,
            pixels: Any,
            proprio: Any,
            *,
            deterministic: bool = False,
            generator: Any | None = None,
        ):
            mean, log_std = self.distribution_parameters(pixels, proprio)
            if deterministic:
                pre_tanh = mean
            else:
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device="cpu",
                    generator=generator,
                ).to(mean.device)
                pre_tanh = mean + log_std.exp() * noise
            squashed = torch.tanh(pre_tanh)
            action = squashed * config.action_magnitude
            normal_log_probability = -0.5 * (
                ((pre_tanh - mean) / log_std.exp()) ** 2
                + 2 * log_std
                + math.log(2 * math.pi)
            )
            log_probability = normal_log_probability.sum(dim=-1, keepdim=True)
            correction = torch.log(
                config.action_magnitude * (1 - squashed.pow(2)) + 1e-6
            ).sum(dim=-1, keepdim=True)
            return action, log_probability - correction

    return Actor()


def _build_critic(torch: Any, nn: Any, config: DsrlConfig):
    encoder = _build_encoder(torch, nn, config)

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            input_dim = config.encoder_latent_dim + 8 + config.action_dim
            self.q_functions = nn.ModuleList(
                [
                    _mlp(
                        nn,
                        input_dim,
                        config.hidden_dims,
                        1,
                        layer_norm=True,
                    )
                    for _ in range(config.num_critics)
                ]
            )

        def forward(self, pixels: Any, proprio: Any, action: Any):
            encoded = self.encoder(pixels)
            inputs = torch.cat((encoded, proprio, action), dim=-1)
            return torch.stack([q(inputs) for q in self.q_functions], dim=0)

    return Critic()


def _build_encoder(torch: Any, nn: Any, config: DsrlConfig):
    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            input_channels = 9
            for output_channels, stride in zip(
                config.encoder_features,
                config.encoder_strides,
                strict=True,
            ):
                layers.extend(
                    (
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=3,
                            stride=stride,
                            padding=0,
                        ),
                        nn.ReLU(),
                    )
                )
                input_channels = output_channels
            self.convolutions = nn.Sequential(*layers)
            with torch.no_grad():
                dummy = torch.zeros(1, 9, config.image_size, config.image_size)
                flattened = int(self.convolutions(dummy).numel())
            self.bottleneck = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, config.encoder_latent_dim),
                nn.LayerNorm(config.encoder_latent_dim),
                nn.Tanh(),
            )

        def forward(self, pixels: Any):
            return self.bottleneck(self.convolutions(pixels))

    return Encoder()


def _mlp(
    nn: Any,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    activate_final: bool = False,
    layer_norm: bool = False,
):
    layers: list[Any] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(previous, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    if activate_final:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _initialize_encoder(nn: Any, encoder: Any) -> None:
    for module in encoder.convolutions:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
    bottleneck = next(
        module for module in encoder.bottleneck if isinstance(module, nn.Linear)
    )
    nn.init.xavier_normal_(bottleneck.weight)
    nn.init.zeros_(bottleneck.bias)


def _initialize_linear_stack(nn: Any, module: Any, *, gain: float = 1.0) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def _initialize_actor(nn: Any, actor: Any) -> None:
    _initialize_encoder(nn, actor.encoder)
    _initialize_linear_stack(nn, actor.trunk)
    nn.init.orthogonal_(actor.mean.weight, gain=1e-2)
    nn.init.zeros_(actor.mean.bias)
    nn.init.orthogonal_(actor.log_std.weight, gain=1e-2)
    nn.init.zeros_(actor.log_std.bias)


def _initialize_critic(nn: Any, critic: Any) -> None:
    _initialize_encoder(nn, critic.encoder)
    for q_function in critic.q_functions:
        _initialize_linear_stack(nn, q_function)


def _random_shift(
    torch: Any,
    pixels: Any,
    *,
    padding: int,
    generator: Any | None = None,
):
    import torch.nn.functional as functional

    batch, _channels, height, width = pixels.shape
    padded = functional.pad(pixels, (padding,) * 4, mode="replicate")
    shifted = []
    offsets = torch.randint(
        0,
        2 * padding + 1,
        (batch, 2),
        device="cpu",
        generator=generator,
    ).to(pixels.device)
    for index in range(batch):
        y = int(offsets[index, 0])
        x = int(offsets[index, 1])
        shifted.append(padded[index, :, y : y + height, x : x + width])
    return torch.stack(shifted, dim=0)


def _resolve_device(torch: Any, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def _isolated_torch_initialization(torch: Any, seed: int):
    """Seed module construction without perturbing another controller's RNG."""

    with _TORCH_INITIALIZATION_LOCK:
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        mps = getattr(torch, "mps", None)
        mps_state = (
            mps.get_rng_state()
            if mps is not None
            and hasattr(mps, "get_rng_state")
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            else None
        )
        try:
            torch.manual_seed(seed)
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            if mps_state is not None:
                mps.set_rng_state(mps_state)


def _optimizer_to(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _resize_rgb(image: np.ndarray, image_size: int) -> npt.NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"DSRL camera image must be HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        if array.dtype.kind not in {"i", "u"} or array.min() < 0 or array.max() > 255:
            raise ValueError("DSRL camera image must contain uint8-compatible RGB")
        array = array.astype(np.uint8)
    source = Image.fromarray(array, mode="RGB")
    current_width, current_height = source.size
    ratio = max(current_width / image_size, current_height / image_size)
    resized_width = int(current_width / ratio)
    resized_height = int(current_height / ratio)
    resized = source.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    padded = Image.new("RGB", (image_size, image_size), 0)
    padded.paste(
        resized,
        (
            max(0, int((image_size - resized_width) / 2)),
            max(0, int((image_size - resized_height) / 2)),
        ),
    )
    return np.asarray(padded, dtype=np.uint8)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_policy_lineage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract immutable PI0 identity without per-request sampler metadata."""

    if not isinstance(metadata, Mapping):
        raise ValueError("hosted PI0 policy metadata must be a mapping")
    missing = [key for key in _PI0_BASE_POLICY_LINEAGE_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            "hosted PI0 policy metadata is missing lineage fields: "
            + ", ".join(missing)
        )
    lineage = {key: metadata[key] for key in _PI0_BASE_POLICY_LINEAGE_KEYS}
    if lineage["base_model"] != "pi0-droid":
        raise ValueError("DSRL requires a pi0-droid base policy")
    if not all(
        isinstance(lineage[key], str) and bool(lineage[key])
        for key in (
            "openpi_config",
            "checkpoint_uri",
            "openpi_source_commit",
            "action_space",
        )
    ):
        raise ValueError("hosted PI0 lineage string fields must be non-empty")
    if lineage["action_horizon"] != 10 or lineage["action_dim"] != 8:
        raise ValueError("DSRL requires the pinned PI0-DROID [10,8] action contract")
    return lineage


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


__all__ = [
    "DSRL_METHOD_VARIANT",
    "DSRL_REFERENCE_COMMIT",
    "DSRL_REFERENCE_REPOSITORY",
    "DsrlConfig",
    "DsrlObservation",
    "DsrlReplayBuffer",
    "TorchDsrlController",
]
