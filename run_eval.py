"""Run DROID simulation rollouts with OpenPI or Cybernetics inference.

Examples:

    python run_eval.py --episodes 10 --headless
    python run_eval.py --backend cybernetics --episodes 10 --headless
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from sim_evals.episode_results import EpisodeResultWriter
from sim_evals.inference.abstract_client import InferenceClient

Backend = Literal["openpi", "cybernetics"]


def _instruction_for_scene(scene: int) -> str:
    instructions = {
        1: "put the cube in the bowl",
        2: "put the can in the mug",
        3: "put banana in the bin",
    }
    try:
        return instructions[scene]
    except KeyError as exc:
        raise ValueError(f"Scene {scene} not supported") from exc


def _create_client(
    backend: Backend,
    *,
    remote_host: str,
    remote_port: int,
    open_loop_horizon: int,
    cybernetics_base_model: str,
    cybernetics_model_path: str | None,
    cybernetics_request_timeout: float,
    cybernetics_session_timeout: float,
) -> InferenceClient:
    if backend == "openpi":
        from sim_evals.inference.droid_jointpos import Client

        return Client(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=open_loop_horizon,
        )

    from sim_evals.inference.cybernetics_dreamzero import Client

    return Client(
        base_model=cybernetics_base_model,
        model_path=cybernetics_model_path,
        request_timeout=cybernetics_request_timeout,
        session_timeout=cybernetics_session_timeout,
        open_loop_horizon=open_loop_horizon,
    )


def _bool_flag(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _error(phase: str, exc: Exception) -> dict[str, str]:
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc)}


def _task_distance(env: Any, scene: int) -> dict[str, Any]:
    entities = {
        1: ("rubiks_cube", "_24_bowl"),
        2: ("_10_potted_meat_can", "_25_mug"),
        3: ("_11_banana", "Magenta_Box"),
    }
    object_name, target_name = entities[scene]
    try:
        object_position = (
            env.unwrapped.scene[object_name]
            .data.root_pos_w[0, :3]
            .detach()
            .cpu()
            .float()
        )
        target_position = (
            env.unwrapped.scene[target_name]
            .data.root_pos_w[0, :3]
            .detach()
            .cpu()
            .float()
        )
        return {
            "object_entity": object_name,
            "target_entity": target_name,
            "object_position_m": object_position.tolist(),
            "target_position_m": target_position.tolist(),
            "center_distance_m": float(
                torch.linalg.vector_norm(object_position - target_position)
            ),
        }
    except Exception as exc:
        return {
            "object_entity": object_name,
            "target_entity": target_name,
            "unavailable": _error("task_distance", exc),
        }


def main(
    episodes: int = 10,
    headless: bool = True,
    scene: int = 1,
    backend: Backend = "openpi",
    results_dir: Path | None = None,
    remote_host: str = "localhost",
    remote_port: int = 8000,
    open_loop_horizon: int = 8,
    cybernetics_base_model: str = "dreamzero-droid",
    cybernetics_model_path: str | None = None,
    cybernetics_request_timeout: float = 2400.0,
    cybernetics_session_timeout: float = 2400.0,
) -> None:
    """Run DROID episodes and write ``episodes.jsonl`` plus ``episodes.json``."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    # Launch Omniverse before importing any IsaacLab-dependent modules.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run DROID policy evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = _instruction_for_scene(scene)
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # A second render cycle loads scene materials correctly.
    client = _create_client(
        backend,
        remote_host=remote_host,
        remote_port=remote_port,
        open_loop_horizon=open_loop_horizon,
        cybernetics_base_model=cybernetics_base_model,
        cybernetics_model_path=cybernetics_model_path,
        cybernetics_request_timeout=cybernetics_request_timeout,
        cybernetics_session_timeout=cybernetics_session_timeout,
    )

    now = datetime.now()
    run_dir = results_dir or Path("runs") / now.strftime("%Y-%m-%d") / now.strftime(
        "%H-%M-%S"
    )
    result_writer = EpisodeResultWriter(run_dir)
    max_steps = env.env.max_episode_length

    try:
        with torch.no_grad():
            for episode_index in range(episodes):
                episode_started = perf_counter()
                video: list[Any] = []
                errors: list[dict[str, str]] = []
                terminated = False
                truncated = False
                steps = 0

                try:
                    obs, _ = env.reset()
                except Exception as exc:
                    errors.append(_error("environment_reset", exc))
                initial_task_distance = _task_distance(env, scene)

                try:
                    client.reset()
                except Exception as exc:
                    errors.append(_error("session_reset", exc))

                if not errors:
                    for _ in tqdm(
                        range(max_steps), desc=f"Episode {episode_index + 1}/{episodes}"
                    ):
                        try:
                            inference = client.infer(obs, instruction)
                        except Exception as exc:
                            errors.append(_error("inference", exc))
                            break

                        if not headless:
                            cv2.imshow(
                                "DROID Cameras",
                                cv2.cvtColor(inference["viz"], cv2.COLOR_RGB2BGR),
                            )
                            cv2.waitKey(1)
                        video.append(inference["viz"])

                        try:
                            action = torch.as_tensor(inference["action"])[None]
                            obs, _, term, trunc, _ = env.step(action)
                        except Exception as exc:
                            errors.append(_error("environment_step", exc))
                            break

                        steps += 1
                        terminated = _bool_flag(term)
                        truncated = _bool_flag(trunc)
                        if terminated or truncated:
                            break

                video_path: str | None = None
                if video:
                    episode_video_path = run_dir / f"episode_{episode_index}.mp4"
                    try:
                        mediapy.write_video(episode_video_path, video, fps=15)
                        video_path = str(episode_video_path)
                    except Exception as exc:
                        errors.append(_error("video_write", exc))

                result = {
                    "episode": episode_index + 1,
                    "backend": backend,
                    "scene": scene,
                    "instruction": instruction,
                    "status": "error" if errors else "completed",
                    "steps": steps,
                    "max_steps": max_steps,
                    "terminated": terminated,
                    "truncated": truncated,
                    "duration_ms": (perf_counter() - episode_started) * 1000,
                    "video_path": video_path,
                    "errors": errors,
                    "inference": client.episode_metrics(),
                    "task_diagnostic": {
                        "protocol": "unscored_upstream_center_distance",
                        "success": None,
                        "initial": initial_task_distance,
                        "final": _task_distance(env, scene),
                    },
                }
                result_writer.record(result)
                tqdm.write(json.dumps(result, sort_keys=True))
    finally:
        try:
            client.close()
        finally:
            try:
                env.close()
            finally:
                if not headless:
                    cv2.destroyAllWindows()
                simulation_app.close()

    print(f"Episode JSONL: {result_writer.jsonl_path}")
    print(f"Episode JSON: {result_writer.json_path}")


if __name__ == "__main__":
    tyro.cli(main)
