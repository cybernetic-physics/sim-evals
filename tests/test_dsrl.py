from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sim_evals.dsrl import (
    DSRL_METHOD_VARIANT,
    DSRL_REFERENCE_COMMIT,
    DsrlConfig,
    DsrlObservation,
    DsrlReplayBuffer,
    TorchDsrlController,
)
from sim_evals.inference.droid_observation import DroidObservation


def _observation(value: int = 0) -> DroidObservation:
    return DroidObservation(
        exterior_image_1_left=np.full((72, 96, 3), value, dtype=np.uint8),
        exterior_image_2_left=np.full((72, 96, 3), value + 1, dtype=np.uint8),
        wrist_image_left=np.full((72, 96, 3), value + 2, dtype=np.uint8),
        joint_position=np.linspace(-1, 1, 7, dtype=np.float32),
        gripper_position=np.asarray([0.25], dtype=np.float32),
        instruction="put the cube in the bowl",
    )


def _small_config() -> DsrlConfig:
    return DsrlConfig(
        image_size=64,
        hidden_dims=(32, 32, 32),
        encoder_latent_dim=16,
        batch_size=2,
        replay_capacity=4,
        random_exploration_episodes=1,
        initial_updates=1,
        updates_per_transition=1,
        random_shift_pixels=0,
        device="cpu",
    )


def _base_policy_metadata() -> dict[str, object]:
    return {
        "base_model": "pi0-droid",
        "openpi_config": "pi0_droid_jointpos_polaris",
        "checkpoint_uri": (
            "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris"
        ),
        "openpi_source_commit": "714ec9aa5e4e9b73b98c6bf3a328f377268e26f9",
        "action_space": "droid_joint_position",
        "action_horizon": 10,
        "action_dim": 8,
        "pi0_initial_flow_noise": {"applied": True, "sha256": "per-request"},
    }


def test_dsrl_observation_concatenates_three_cameras_and_normalizes_state() -> None:
    encoded = DsrlObservation.from_droid(_observation(3), image_size=32)

    assert encoded.pixels.shape == (32, 32, 9)
    assert encoded.pixels.dtype == np.uint8
    assert encoded.proprio.shape == (8,)
    assert encoded.proprio.dtype == np.float32
    assert np.isfinite(encoded.proprio).all()
    np.testing.assert_array_equal(encoded.pixels[0, 0], np.zeros(9, dtype=np.uint8))
    np.testing.assert_array_equal(encoded.pixels[16, 16, :3], [3, 3, 3])
    np.testing.assert_array_equal(encoded.pixels[16, 16, 3:6], [4, 4, 4])
    np.testing.assert_array_equal(encoded.pixels[16, 16, 6:], [5, 5, 5])
    np.testing.assert_allclose(
        encoded.proprio,
        np.concatenate(
            (
                np.linspace(-1, 1, 7, dtype=np.float32),
                np.asarray([0.25], dtype=np.float32),
            )
        ),
    )


def test_replay_is_bounded_and_owns_chunk_discount() -> None:
    config = _small_config()
    replay = DsrlReplayBuffer(config)
    first = DsrlObservation.from_droid(_observation(1), image_size=config.image_size)
    second = DsrlObservation.from_droid(_observation(2), image_size=config.image_size)
    action = np.zeros(32, dtype=np.float32)

    for index in range(6):
        replay.insert(
            observation=first,
            action=action + index,
            reward=-1.0,
            discount=config.gamma**10,
            mask=1.0,
            next_observation=second,
        )

    assert replay.size == 4
    assert replay.cursor == 2
    batch = replay.sample(2, np.random.default_rng(7))
    assert batch["observations"].shape == (2, 64, 64, 9)
    assert batch["actions"].shape == (2, 32)


def test_config_and_replay_reject_invalid_controller_state() -> None:
    with pytest.raises(ValueError, match="hidden_dims"):
        DsrlConfig(hidden_dims=())

    config = _small_config()
    replay = DsrlReplayBuffer(config)
    observation = DsrlObservation.from_droid(
        _observation(1), image_size=config.image_size
    )
    with pytest.raises(ValueError, match="finite"):
        replay.insert(
            observation=observation,
            action=np.full(32, np.nan, dtype=np.float32),
            reward=-1.0,
            discount=config.gamma**10,
            mask=1.0,
            next_observation=observation,
        )


def test_torch_controller_updates_checkpoints_and_resumes(tmp_path) -> None:
    pytest.importorskip("torch")
    controller = TorchDsrlController(_small_config())
    observation = _observation(4)
    next_observation = _observation(5)
    action = controller.select_action(observation)
    assert action.shape == (32,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()

    transition = SimpleNamespace(
        observation=observation,
        next_observation=next_observation,
        dsrl_action=action,
        reward=-1.0,
        discount=controller.gamma**10,
        mask=1.0,
        policy_metadata=_base_policy_metadata(),
    )
    first_metrics = controller.record_transition(transition)
    second_metrics = controller.record_transition(transition)
    assert first_metrics["updates"] == 0
    assert second_metrics["updates"] == 0
    training_metrics = controller.train_after_trajectory(2)
    assert training_metrics["updates"] == 1
    assert training_metrics["trajectory_transitions"] == 2
    assert controller.training_trajectories == 1
    assert controller.trained_transitions == 2

    base_metadata = {
        key: value
        for key, value in _base_policy_metadata().items()
        if key != "pi0_initial_flow_noise"
    }
    checkpoint_dir = tmp_path / "controller" / "checkpoint-000001"
    manifest = controller.save_checkpoint(
        checkpoint_dir,
        base_policy_metadata=base_metadata,
    )
    assert manifest["method"] == DSRL_METHOD_VARIANT
    assert manifest["reference_commit"] == DSRL_REFERENCE_COMMIT
    assert len(manifest["checkpoint_sha256"]) == 64
    assert len(manifest["replay_sha256"]) == 64

    expected = controller.select_action(observation, deterministic=True)
    resumed = TorchDsrlController.load_checkpoint(
        checkpoint_dir,
        device="cpu",
        expected_base_policy_metadata=base_metadata,
    )
    actual = resumed.select_action(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert resumed.transitions == controller.transitions
    assert resumed.updates == controller.updates
    assert resumed.training_trajectories == controller.training_trajectories
    assert resumed.trained_transitions == controller.trained_transitions
    assert resumed.replay.size == controller.replay.size


def test_trajectory_is_fully_inserted_before_any_updates() -> None:
    pytest.importorskip("torch")
    controller = TorchDsrlController(_small_config())
    observation = _observation(10)
    transition = SimpleNamespace(
        observation=observation,
        next_observation=_observation(11),
        dsrl_action=controller.select_action(observation),
        reward=-1.0,
        discount=controller.gamma**10,
        mask=1.0,
        policy_metadata=_base_policy_metadata(),
    )

    for _ in range(3):
        metrics = controller.record_transition(transition)
        assert metrics["updates"] == 0
    assert controller.replay.size == 3

    metrics = controller.train_after_trajectory(3)
    assert metrics["trajectory_updates"] == 1
    assert metrics["updates"] == 1


def test_replay_samples_with_replacement_below_batch_size() -> None:
    config = DsrlConfig(
        image_size=32,
        hidden_dims=(8,),
        encoder_latent_dim=4,
        batch_size=8,
        replay_capacity=2,
        initial_updates=1,
        updates_per_transition=1,
        random_shift_pixels=0,
        device="cpu",
    )
    replay = DsrlReplayBuffer(config)
    observation = DsrlObservation.from_droid(
        _observation(1), image_size=config.image_size
    )
    replay.insert(
        observation=observation,
        action=np.zeros(32, dtype=np.float32),
        reward=-1.0,
        discount=config.gamma**10,
        mask=1.0,
        next_observation=observation,
    )

    batch = replay.sample(config.batch_size, np.random.default_rng(3))
    assert batch["actions"].shape == (8, 32)


def test_torch_controllers_own_independent_exploration_rng() -> None:
    pytest.importorskip("torch")
    observation = _observation(7)
    primary = TorchDsrlController(_small_config())
    first = primary.select_action(observation)
    TorchDsrlController(_small_config())
    second = primary.select_action(observation)

    reference = TorchDsrlController(_small_config())
    expected_first = reference.select_action(observation)
    expected_second = reference.select_action(observation)
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second, expected_second)


def test_torch_checkpoint_loader_rejects_corruption(tmp_path) -> None:
    pytest.importorskip("torch")
    controller = TorchDsrlController(_small_config())
    observation = _observation(8)
    transition = SimpleNamespace(
        observation=observation,
        next_observation=_observation(9),
        dsrl_action=controller.select_action(observation),
        reward=-1.0,
        discount=controller.gamma**10,
        mask=1.0,
        policy_metadata=_base_policy_metadata(),
    )
    controller.record_transition(transition)
    checkpoint_dir = tmp_path / "checkpoint"
    controller.save_checkpoint(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "controller.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="hash"):
        TorchDsrlController.load_checkpoint(checkpoint_dir, device="cpu")


def test_training_resume_rejects_lightweight_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    controller = TorchDsrlController(_small_config())
    observation = _observation(12)
    transition = SimpleNamespace(
        observation=observation,
        next_observation=_observation(13),
        dsrl_action=controller.select_action(observation),
        reward=-1.0,
        discount=controller.gamma**10,
        mask=1.0,
        policy_metadata=_base_policy_metadata(),
    )
    controller.record_transition(transition)
    checkpoint_dir = tmp_path / "latest"
    manifest = controller.save_checkpoint(checkpoint_dir, include_replay=False)
    assert manifest["replay"] is None

    with pytest.raises(ValueError, match="replay-bearing"):
        TorchDsrlController.load_checkpoint(
            checkpoint_dir,
            device="cpu",
            require_replay=True,
        )

    inspection_controller = TorchDsrlController.load_checkpoint(
        checkpoint_dir,
        device="cpu",
    )
    assert inspection_controller.replay.size == 0
