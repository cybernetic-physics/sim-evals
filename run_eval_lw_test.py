import tyro
import argparse
import gymnasium as gym
import torch
import numpy as np
import mediapy
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main(
        environment: str,
        run_folder: Path,
        episodes: int = 3,
        num_envs: int = 1,
        headless: bool = False,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Initialize the env
    env_cfg = parse_env_cfg(
        environment,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=True,
    )
    env = gym.make(environment, cfg=env_cfg)

    # Resume CSV logging
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "progress": pd.Series(dtype="float"),
            }
        )
    next_ep_id = len(episode_df)
    if next_ep_id >= episodes:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    obs, info = env.reset()
    instruction = info["instruction"]
    print(f"Instruction: {instruction}")

    max_steps = env.unwrapped.max_episode_length

    # Per-env state
    videos = [[] for _ in range(num_envs)]
    ep_steps = np.zeros(num_envs, dtype=np.int32)
    # Map each env slot to the episode id it's currently running
    env_ep_ids = np.arange(next_ep_id, next_ep_id + num_envs)

    print(f" >>> Starting eval from episode {next_ep_id + 1} of {episodes} ({num_envs} parallel envs) <<< ")
    bar = tqdm(total=episodes - next_ep_id, desc="Episodes completed")

    with torch.no_grad():
        while True:
            env.unwrapped.sim.render()
    bar.close()
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
