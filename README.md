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

### DROID Policies through Cybernetics

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
`ServiceClient.create_sampling_client(base_model="dreamzero-droid")` or
`base_model="pi0-droid"` returns a sampling client with the typed DROID helper:

```python
observation = cybernetics.DroidObservation.from_numpy(...)
future = sampler.sample_droid(observation)
response = future.result(timeout=2400)
```

The observation carries both exterior cameras, the wrist camera, seven arm
joint positions, one gripper position, and the natural-language instruction.
The response exposes a robot-space `action_chunk` with shape `[1, N, 8]`. Use an
eight-action open-loop cadence for DreamZero and ten for the PolaRiS PI0
joint-position policy. Each episode gets a fresh sampling session and the
previous session is cancelled so backend policy state is released.

Hosted evidence schema v5 preserves both the full normalized model output in
`sampled_action_chunk` and the configured open-loop execution slice in
`action_chunk`. `sampled_action_chunk_shape` records the normalized `[N, 8]`
shape before horizon or maximum-step truncation.

The `flatdict` build override in `pyproject.toml` pins its isolated build to a
setuptools release that still provides the undeclared `pkg_resources` module
required by IsaacLab's dependency.

### Hosted Cybernetic Physics DROID E2E

`run_hosted_eval.py` keeps simulation and policy execution on their intended
planes:

- Cybernetic Physics launches the configured environment and owns every
  `isaac.*` MCP scene read, camera capture, joint write, and simulation step.
- Worldlines owns policy execution. The runner calls the public Cybernetics
  `sample_droid` helper and never loads a policy into Isaac.

Use a Cybernetics SDK release that provides
`cybernetics.sim.SimulationClient.mcp_session`. Authenticate with the normal
SDK login or environment-based credential flow; this runner deliberately has
no credential arguments. The environment URI can be passed directly or through
`CYBERNETICS_DROID_ENV_URI`:

```bash
export CYBERNETICS_DROID_ENV_URI=cybernetics://envs/ENV_ID/versions/VERSION_ID
python run_hosted_eval.py --max-action-steps 450
```

For the inference-only PI0 joint-position policy, record a 100-action rollout
and intentionally stop the launched simulator after the MP4 is durable:

```bash
python run_hosted_eval.py \
  --base-model pi0-droid \
  --environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --open-loop-horizon 10 \
  --max-action-steps 100 \
  --record-video \
  --video-fps 15 \
  --stop-session \
  --results-dir "$PWD/runs/hosted-production/pi0-droid-scene1"
```

PI0 does not support `--policy-mode sde` or `--include-predicted-video`.

On macOS, use an isolated environment because the local IsaacLab dependencies
are Linux-only. Point `CYBERNETICS_SDK_CHECKOUT` at a current Cybernetics SDK
source checkout. This command runs only the hosted client and Pillow, and uses
a caller-selected run directory so the evidence path is stable and easy to
archive:

```bash
cd ~/wagmi/sim-evals
export CYBERNETICS_SDK_CHECKOUT=/path/to/cybernetic-physics/sdk/python
PYTHONPATH="$PWD/src" \
CYBERNETICS_BASE_URL=https://api.cyberneticphysics.com \
uv run --isolated --no-project \
  --with "$CYBERNETICS_SDK_CHECKOUT" \
  --with pillow \
  python "$PWD/run_hosted_eval.py" \
  --environment-uri cybernetics://envs/ENV_ID/versions/VERSION_ID \
  --instruction "put the cube in the bowl" \
  --max-action-steps 450 \
  --open-loop-horizon 8 \
  --results-dir "$PWD/runs/hosted-droid/droid-replication"
```

Normal accounts use the shared warm pool and should omit `--runtime-provider`.
For an operator acceptance run, a service or system-admin credential may add
`--runtime-provider vast` to request dedicated Cybernetic Physics simulation
capacity while keeping scene launch, MCP, and teardown under the same SDK.

A cold dedicated host can outlive a client-side launch timeout while the
control plane continues provisioning the same session. Resume that session
without renting duplicate capacity:

```bash
python run_hosted_eval.py \
  --environment-uri cybernetics://envs/ENV_ID/versions/VERSION_ID \
  --session-id sess_EXISTING \
  --launch-timeout-seconds 2700 \
  --results-dir "$PWD/runs/hosted-droid/resumed-session"
```

An attached session is caller-owned, so the evaluator never stops it. Stop it
through the Cybernetics SDK after preserving the evidence. Sessions launched by
the evaluator are also retained by default so the live scene remains available
for inspection after the rollout. Pass `--stop-session` only when teardown is
intentional. The older explicit `--keep-session` spelling remains accepted.

Omit `--results-dir` to create a collision-resistant UTC directory such as
`runs/hosted-droid/20260712T140506.123456Z`. The final console JSON reports the
directory used. A selected directory represents one run slot; rerunning into
the same path replaces the known result, error, and frame evidence files while
leaving unrelated files alone.

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
validates or reloads the DROID articulation, creates a fresh evaluator-owned
generation of exterior/wrist cameras, captures RGB PNG artifacts, reads named
joint positions, samples one policy action chunk, and applies the configured
open-loop slice before observing again. Fresh camera paths avoid stale path-keyed render
products on older hosted images; the extension also releases those cached
wrappers when cameras are replaced or deleted. The runner selects the first
external camera for the streamed viewer through `isaac.set_active_camera`, with
an `isaac.execute_script` compatibility fallback for older hosted images. This
viewer selection does not change any DROID observation camera pose or pixels.
Explicit pre-dispatch `BRIDGE_OFFLINE` and `ISAAC_UNREACHABLE` responses are
retried with the configured readiness poll interval. Other command failures are
not replayed, so ambiguous or application-level errors still fail closed. An
HTTP 502 is replayed only for absolute `isaac.set_joint_positions` targets,
where sending the identical target again is idempotent; non-idempotent simulation
steps still fail closed on transport ambiguity.
An action is appended to evidence only after `step_simulation` reports the full
requested frame count without `timed_out`; partial steps remain failure evidence,
not acknowledged actions. Before creating a fresh camera generation, the runner
deletes older evaluator-owned external roots and wrist cameras. The updated
extension releases their render products, keeping retained sessions bounded.
Sessions launched successfully by the evaluator remain running after the
rollout. Use `--stop-session` to opt into cleanup. Attached sessions always
remain caller-owned.

The runner reads `physics_dt` from `isaac.get_simulation_state` and derives the
integer number of physics updates nearest to 15 Hz. It requests play immediately
before each bounded `step_simulation` call and pause immediately afterward, then
keeps policy inference and camera capture paused. `runtime.json` and every
applied-action record preserve the measured cadence and before/after simulation
time, exposing any timeline drift between commands. Use `--physics-steps-per-action` only
for a deliberate override. Gripper actions match the reference environment:
values greater than `0.5` close to `pi/4`; all other values open to zero.

Before this retained-by-default contract, omitting `--keep-session` caused a
successful evaluator to call `SimulationClient.stop_session()` in its cleanup
path. The platform then correctly displayed the session as `TERMINATED` with
`stop_reason=user_stopped`; that state was evaluator-requested teardown, not an
Isaac crash. Result evidence now includes `session_retained` so lifecycle intent
is visible beside the session id.

The runner writes `config.json` before launching the hosted session. During
each observation it preserves the exact PNG bytes already downloaded through
MCP for all three DROID camera roles. Captures must be 640x360 and contain
meaningful luminance variance plus both non-dark and non-white content; black,
blank, clipped-sliver, and render-race frames retry and then fail closed before
the policy is called. The runner then atomically writes one terminal
record: `result.json` after success or `error.json` after failure. Errors include
the exception type and message, plus the hosted session ID when launch reached
that point. The evidence layout is:

```text
<results-dir>/
|-- config.json
|-- actions.jsonl                       # samples, accepted targets, applied actions
|-- result.json                         # success only
|-- error.json                          # failure only
|-- rollout.mp4                         # post-action exterior-camera rollout
|-- runtime.json                        # measured physics dt and control cadence
|-- frames/
    |-- sample-00000-exterior-1.png
    |-- sample-00000-exterior-2.png
    |-- sample-00000-wrist.png
    `-- sample-00001-*.png               # one triplet per later observation
|-- video-frames/
    |-- action-00000.png                 # lossless MP4 source frame per action
    `-- manifest.json                    # frame hashes and selected camera
`-- samples/
    |-- sample-00000-predicted-video.npy # when requested
    `-- sample-00000-trajectory.npz      # SDE tensor-map artifacts
```

`config.json`, `result.json`, `error.json`, and every JSONL record are versioned
and machine-readable. Each sample record contains the full sampled action chunk
and bounded execution slice. An `action_target` record is privately written and
`fsync`'d after Isaac accepts the transformed joint positions and indices;
`applied_action` repeats that exact target only after the configured physics
steps complete. Both records retain the original policy action. A failure thus
distinguishes accepted targets from fully stepped actions while retaining any
frames or tensor artifacts already written.

PI0 sample records also require and preserve the loaded policy provenance:
OpenPI source commit, config, checkpoint URI, action space, horizon, and action
dimension. A missing or mismatched profile fails before any action is applied.
The MP4 manifest records H.264 codec, FPS, dimensions, duration, byte length,
SHA-256, source camera, frame count, and the lossless source-frame manifest.
Video support is checked before any hosted session work begins. The runner uses
`mediapy` when installed and otherwise falls back to local `ffmpeg` plus
`ffprobe`, including a complete decode and frame-count validation.

If an older runner completed the policy/simulator loop but failed while encoding
the MP4, recover the already durable post-action PNGs without moving the robot
again:

```bash
python recover_hosted_video.py runs/hosted-production/<RUN_DIRECTORY>
```

Recovery writes `rollout.mp4`, `video-frames/manifest.json`, and
`video-recovery.json`. It deliberately preserves the original `error.json`
instead of relabeling the interrupted run as an uninterrupted evaluator success.

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
