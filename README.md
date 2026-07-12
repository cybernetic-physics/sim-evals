# DROID Sim Evaluation

This repository contains scripts for evaluating DROID policies in a simple ISAAC Sim environment.

Here is an example rollout of a pi0-FAST-DROID policy:

Scene 1

![Scene 1](./docs/scene1.gif)

Scene 2

![Scene 2](./docs/scene2.gif)

Scene 3

![Scene 3](./docs/scene3.gif)

The simulation is tuned to work *zero-shot* with DROID policies trained on the real-world DROID dataset, so no separate simulation data is required.

**Note:** The current simulator works best for policies trained with *joint position* action space (and *not* joint velocity control). We provide examples for evaluating pi0-FAST-DROID policies trained with joint position control below.


## Installation

Clone the repo
```bash
git clone --recurse-submodules git@github.com:arhanjain/sim-evals.git
cd sim-evals
```

Install uv (see: https://github.com/astral-sh/uv#installation)

For example (Linux/macOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate virtual environment
```bash
uv sync
source .venv/bin/activate
```

## Quick Start

First, make sure you download the simulation assets into the root of this directory
```bash
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

Then, in a separate terminal, launch the policy server on `localhost:8000`. 
For example, to launch a pi0-FAST-DROID policy (with joint position control),
checkout [openpi](https://github.com/Physical-Intelligence/openpi) and use the `polaris` configs 
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

**Note**: We set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to avoid JAX hogging all the GPU memory (incase Isaac Sim is using the same GPU).

Finally, run the evaluation script:
```bash
python run_eval.py --episodes [INT] --scene [INT] --headless
```

Each run writes one result object per episode to both `episodes.jsonl` and
`episodes.json` alongside the episode videos under `runs/`. Results include the
selected backend, termination state, inference latency summary, and structured
errors. The upstream environments define only a 30-second timeout, not a task
success term. Results therefore leave `success` unset and record the start/end
object-to-target center distance as a diagnostic rather than inventing a pass
threshold.

### DreamZero-DROID through Cybernetics

Install the Cybernetics SDK, configure its normal environment-based
authentication, and run:

```bash
python run_eval.py --backend cybernetics --episodes 10 --scene 1 --headless
```

On a multi-GPU evaluator, select the Isaac physics device explicitly. This is
useful when another policy server already owns the default GPU:

```bash
python run_eval.py --backend cybernetics --episodes 10 --scene 1 --headless --device cuda:1
```

No credential is accepted as a command-line argument. The integration assumes
`ServiceClient.create_sampling_client(base_model="dreamzero-droid")` returns a
sampling client with the typed DROID helper:

```python
observation = cybernetics.DroidObservation.from_numpy(...)
future = sampler.sample_droid(observation)
response = future.result(timeout=2400)
```

The observation carries both exterior cameras, the wrist camera, seven arm
joint positions, one gripper position, and the natural-language instruction.
The response exposes a robot-space `action_chunk` with shape `[1, N, 8]`; the
evaluation follows DreamZero's eight-action open-loop cadence. Each episode gets
a fresh sampling session and the previous session is cancelled so backend frame
history and causal cache ownership are released.

The `flatdict` build override in `pyproject.toml` pins its isolated build to a
setuptools release that still provides the undeclared `pkg_resources` module
required by IsaacLab's dependency.

### Hosted Cybernetic Physics DROID E2E

`run_hosted_eval.py` keeps simulation and policy execution on their intended
planes:

- Cybernetic Physics launches the configured environment and owns every
  `isaac.*` MCP scene read, camera capture, joint write, and simulation step.
- DGX Spark is the sole DreamZero execution plane. The runner calls the public
  Cybernetics `sample_droid` helper and does not load DreamZero into Isaac.

Use a Cybernetics SDK release that provides
`cybernetics.sim.SimulationClient.mcp_session`. Authenticate with the normal
SDK login or environment-based credential flow; this runner deliberately has
no credential arguments. The environment URI can be passed directly or through
`CYBERNETICS_DROID_ENV_URI`:

```bash
export CYBERNETICS_DROID_ENV_URI=cybernetics://envs/ENV_ID/versions/VERSION_ID
python run_hosted_eval.py --max-action-steps 450
```

The environment bundle should contain the three task scenes and the DROID USD
at `/data/workspace/franka_robotiq_2f_85_flattened.usd`. Override the latter
when the bundle uses another workspace path:

```bash
python run_hosted_eval.py \
  --environment-uri cybernetics://envs/ENV_ID/versions/VERSION_ID \
  --robot-usd-path /data/workspace/assets/droid.usd \
  --instruction "put the can in the mug"
```

After the hosted session reaches the running state, the runner opens its MCP
session and polls `isaac.get_scene_info` until the extension is ready. It then
validates or reloads the DROID articulation, creates any missing exterior/wrist
cameras, captures RGB PNG artifacts, reads named joint positions, samples one
DreamZero action chunk, and applies up to eight actions before observing again.
Sessions are stopped on success or failure unless `--keep-session` is set.

## Minimal Example

```python
env_cfg.set_scene(scene) # pass scene integer
env = gym.make("DROID", cfg=env_cfg)

obs, _ = env.reset()
obs, _ = env.reset() # need second render cycle to get correctly loaded materials
client = # Your policy of choice

max_steps = env.env.max_episode_length
for _ in tqdm(range(max_steps), desc=f"Episode"):
    action = client.infer(obs, INSTRUCTION) # calling inference on your policy
    action = torch.tensor(ret["action"])[None]
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
env.close()
```
