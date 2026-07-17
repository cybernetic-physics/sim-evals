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
eight-action open-loop cadence for both DreamZero and the PolaRiS PI0
joint-position policy. Each episode gets a fresh sampling session and the
previous session is cancelled so backend policy state is released.

Hosted evidence schema v9 preserves both the full normalized model output in
`sampled_action_chunk` and the configured open-loop execution slice in
`action_chunk`. `sampled_action_chunk_shape` records the normalized `[N, 8]`
shape before horizon or maximum-step truncation. Opt-in task acceptance also
stores a bounded PhysX contact trace for every action update, including signed
separation, normal and friction impulses, and contact points.
DSRL runs additionally record
the verified steering noise, sparse chunk transition, and controller counters
in `dsrl-transitions.jsonl`.

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

Before sampling, the hosted runner inventories the robot's joints through a
USD-only script that cannot initialize an articulation or physics world. It then
stops the timeline, applies the `cybernetics_droid_contact_v1` physics profile:
`400/80` arm drive gains, Panda effort and velocity limits, a 1 rad/s gripper
limit, 64/1 articulation solver iterations, disabled articulation-link gravity,
3 m/s maximum depenetration velocity, 240 Hz physics, TGS, scene and cube CCD,
late articulation-contact solving, and explicit `2 mm/0 mm` contact/rest
offsets on task colliders. The physics
context is explicitly reinitialized after that profile is complete, so later
tensor views cannot retain stale link metadata. The first exact frame then
commits the new context before the runtime-only joint read. The runner also
restores the benchmark arm pose with an open gripper. Fixed play-every-frame
timeline settings and a matching 240 Hz minimum simulation rate make each app
update one physics substep. Policy actions use an atomic sixteen-substep MCP
call that ends paused and must advance exactly one 15 Hz control interval;
timeline drift fails the run instead of silently holding targets too long.
The repository's local IsaacLab DROID environment remains a 120 Hz reference;
240 Hz is the hosted contact-hardening profile, not a claim that rate alone
causes better contact behavior.

The runtime preflight fails closed unless the gripper drive remains at the
benchmark's `100/0.0002/16.5` stiffness, damping, and angular-effort values and
the cube has the pinned `0.04 kg` mass. It replaces the historical effective
friction coefficient of `10` with dedicated finger (`1.5/1.2`), cube
(`0.8/0.6`), bowl (`0.6/0.5`), and table (`0.5/0.4`) static/dynamic materials,
all using `average` combination. The source cube payload does not author a
positive mass, so the runner authors the value before the single physics
rebuild and verifies the composed runtime. It archives exact collider paths,
approximation tokens, offsets, material bindings, and drive values in
`runtime.json`.

Task-acceptance runs require runtime articulation provenance for both DOF order
and every joint observation and target. Authored USD drive targets are not
accepted as measured robot state or as an actuation fallback. Before the first
policy observation, the commanded
benchmark pose must remain within the configured arm/gripper tolerances for two
consecutive control intervals. Camera retries occur while paused and never add
unrecorded physics steps between an applied action and its task-state evidence.
Task acceptance requires complete per-substep contact manifolds for both
finger/cube pairs and the cube/receptacle pair. It rejects penetration above
`1 mm`, a per-contact normal impulse above `0.5 N*s`, non-opposing bilateral
finger normals, sparse contact during a claimed lift, and more than `1 cm` of
gripper-relative object slip while the close command remains active. Placement
is causal only after a post-lift open command and measured bowl contact.

Before accepting a manipulation episode, prove the hosted runtime can
distinguish a clean separated control from penetration, excessive impulse, and
a saturated contact buffer through the same public SDK and session-scoped MCP
path used by the evaluator:

```bash
uv run --isolated --no-project \
  --with /path/to/cybernetic-physics/sdk/python \
  python run_contact_integrity_smoke.py \
  --environment-uri cybernetics://envs/ENV_ID/versions/VERSION_ID \
  --runtime-provider warm_pool
```

Use `--session-id SESSION_ID --keep-session` to validate an existing session
before running `run_hosted_eval.py` against that same runtime. The smoke command
creates rigid bodies only beneath `/World/ContactIntegritySmoke`, removes them
before returning, saves every raw trace under `runs/contact-integrity/`, and
exits nonzero unless all four machine-verifiable controls behave as expected.
Video alone is never accepted as contact-integrity evidence.

Use a Cybernetics SDK release that provides
`cybernetics.sim.SimulationClient.mcp_session`. Authenticate with the normal
SDK login or environment-based credential flow; this runner deliberately has
no credential arguments. Its MCP context explicitly requests the SDK maximum
`86400`-second TTL for long rollouts, while keeping the credential session scoped;
exiting the context revokes it. The environment URI can be passed directly or
through `CYBERNETICS_DROID_ENV_URI`:

```bash
export CYBERNETICS_DROID_ENV_URI=cybernetics://envs/ENV_ID/versions/VERSION_ID
python run_hosted_eval.py --max-action-steps 450
```

For the inference-only PI0 joint-position policy, run the full scene-1 action
cap and stop early only when the policy has lifted the cube, released it inside
the bowl, and left it settled for three consecutive checks:

```bash
python run_hosted_eval.py \
  --base-model pi0-droid \
  --environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --task-success-predicate scene1-cube-in-bowl \
  --open-loop-horizon 8 \
  --max-action-steps 450 \
  --record-video \
  --video-fps 15 \
  --stop-session \
  --results-dir "$PWD/runs/hosted-production/pi0-droid-scene1"
```

PI0 does not support `--policy-mode sde` or `--include-predicted-video`.

To compare physics profiles without changing policy output, replay only the
verified applied-action prefix from a completed schema-v9 PI0 evidence
directory:

```bash
python run_hosted_eval.py \
  --base-model pi0-droid \
  --environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --task-success-predicate scene1-cube-in-bowl \
  --replay-evidence-dir "$PWD/runs/hosted-production/SOURCE_RUN" \
  --physics-hz 240 \
  --solver-position-iterations 64 \
  --solver-velocity-iterations 1 \
  --stop-session \
  --results-dir "$PWD/runs/hosted-production/pi0-droid-replay"
```

Replay preflight verifies a manifest-last inventory of every source artifact,
the schema and producer provenance, PI0 profile, environment, task and camera
contracts, full sample chunks, sequential sample/chunk mapping, derived runtime
joint targets, complete task-state coverage, source runtime cadence, and
continuous bounded-drift action timing before launching. The fresh runtime must
settle within `0.005 rad` per arm joint and `0.01` gripper units of the source
robot state. Initial cube/bowl bounds, runtime position, and gripper-reference
geometry must remain within `0.001 m`; initial task velocities must remain within
`0.01` and retain the same measurement provenance. Physics rate and solver
iterations may differ deliberately for a controlled comparison.

This is a manifest-bound action-prefix replay under bounded initial-state
variation, not deterministic episode reproduction. Simulator scheduling,
contact solver internals, and other hidden state are not captured, so a replay
can show that one profile behaved better for the same verified action prefix;
it cannot prove that a single changed parameter caused the difference. Replay
never constructs a Worldlines client, cannot attach to an existing scene, and
requires evaluator-owned session cleanup. An ambiguous
`isaac.step_simulation` transport failure is never retried because doing so
could apply one policy action twice.

The task predicate is opt-in. Without it, completion continues to mean only
that the requested policy actions were transported and applied. The predicate
uses read-only USD bounds and the Isaac 6 physics tensor view for rigid-body
velocity queries, with Dynamic Control retained only as a legacy fallback. It
records the velocity source in every task-state record and fails closed when
neither backend can return measured velocity; it never moves the cube or bowl.
It requires two consecutive lifted states with measured finger closure above
`0.25`, while release remains at or below `0.20`. The hysteresis recognizes a
real obstructed grasp: the scene-1 cube holds the normalized finger joint near
`0.34`, whereas an empty fully closed gripper reaches `1.0`. The predicate
rejects object jumps above the per-action motion bound, uses a
conservative fraction of the bowl's world bounds, and requires observed release
after a post-lift open command.

Task acceptance additionally requests three exact contact pairs in the same
atomic step as each policy action: left finger/cube, right finger/cube, and
cube/receptacle. A lift requires bilateral finger support at the end of the
action. Placement requires receptacle support, and a support loss greater than
1 cm while the close command remains active permanently invalidates the run.
The provisional scene-1 penetration ceiling is 1 mm; exceeding it, filling a
contact buffer, dropping a pair, emitting non-finite data, or omitting any
update fails closed. This is an evaluator guardrail, not a calibrated
real-hardware tolerance. Replace it only after paired collision-geometry and
system-identification measurements. Placement also requires stable cube and
bowl velocity. Missing PhysX velocity fails closed. Direct placement, a moving
receptacle, or a still-closed gripper cannot pass. The final
predicate read occurs after video capture so camera-recovery physics steps
cannot invalidate an already-recorded verdict.

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

### PI0-DROID DSRL training

Install the optional PyTorch controller dependency, then start with one bounded
canary against an exact immutable environment version:

```bash
uv sync --extra dsrl
python run_hosted_dsrl.py \
  --environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --episodes 1 \
  --max-action-steps 200 \
  --results-dir "$PWD/runs/hosted-dsrl/canary"
```

This is a trajectory-wise port of the public DSRL PI0 real-robot loop. The
hosted PI0-DROID policy stays frozen. A local pixel-SAC controller emits one
32-dimensional initial-flow-noise action, repeated across PI0's 10-step flow
horizon. The first complete trajectory uses Gaussian exploration. Every chunk
transition is held until the fresh simulation and sampling session have been
torn down, then the complete trajectory is inserted into replay before any
optimization. Training performs 5,000 updates after the first trajectory and
`trajectory_chunks * 30` updates thereafter by default.

The sparse chunk reward is `-1` until the strict settled cube-in-bowl predicate
passes and `0` on success. A one-episode/200-primitive-action canary and video
recording off are the safe defaults. More than one sparse-reward training
episode requires an explicit `--allow-zero-success-training` acknowledgement or
an easier curriculum; do not treat that acknowledgement as evidence that the
reward is informative.

Resource overrides fail before controller allocation above 8,192 replay
transitions, 50,000 initial updates, or 512 updates per later transition. These
are safety ceilings, not recommended training settings; the pinned defaults
remain 2,048, 5,000, and 30 respectively.

The hosted method is intentionally named
`dsrl_pixels_proprio_no_vlm_token_v1`: the public sampler does not expose PI0's
final 2,048-dimensional VLM token. It also uses a bounded replay ring and omits
the reference color jitter while retaining random edge-padded shifts. The
bounded MPS canary defaults to batch size 16 and seed 42 rather than the
reference real-robot launcher's batch size 256 and seed 0. These deviations are
persisted in controller and evidence metadata.

Checkpoint roles are deliberately different:

- `controller/latest` is a replay-free weights/optimizer artifact for inspection
  or evaluation. It is not an exact training-resume point, and `--resume`
  rejects it.
- `controller/checkpoint-NNNNNN` contains replay and is written at the configured
  interval and at the final training episode. Resume training only from one of
  these replay-bearing directories.

Run deterministic promotion evidence from a frozen replay-bearing checkpoint
without collecting replay or updating/checkpointing the controller:

```bash
python run_hosted_dsrl.py \
  --environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --episodes 0 \
  --eval-episodes 1 \
  --resume "$PWD/runs/hosted-dsrl/train/controller/checkpoint-000010" \
  --record-video \
  --results-dir "$PWD/runs/hosted-dsrl/promotion-eval"
```

Evaluation-only mode requires both `--resume` and at least one eval episode. It
accepts only a replay-bearing checkpoint, so a successful video can be tied to
an existing trained controller rather than one mutated immediately beforehand.

Each episode creates a fresh PI0 sampling session and a fresh owned simulation
session. The runner also verifies the pinned PI0 checkpoint lineage and the
SHA-256 acknowledgement of the exact `[10, 32]` noise tensor before applying
robot actions.

### Deterministic scene curriculum

Plan train, validation, and held-out variants without launching any workflow:

```bash
python run_curriculum.py \
  --base-environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --train-variants 8 \
  --validation-variants 4 \
  --held-out-variants 4 \
  --manifest "$PWD/runs/droid-curriculum/manifest.json" \
  --dry-run
```

The plan requires an exact `cybernetics://envs/env_.../versions/ver_...` source,
uses unique deterministic seeds, and writes a SHA-256-pinned manifest. It varies
reachable initial geometry, cameras, lighting, colors, and non-occluding
distractors while requiring dynamics, robot state, task paths, and acceptance
semantics to remain unchanged.

Launching is an explicit side effect. Reuse the reviewed manifest and create at
most one workflow in the invocation:

```bash
python run_curriculum.py \
  --base-environment-uri "$CYBERNETICS_DROID_ENV_URI" \
  --manifest "$PWD/runs/droid-curriculum/manifest.json" \
  --launch \
  --max-launches 1
```

Launches are sequential. The manifest records each workflow id immediately,
resumes an existing run instead of duplicating it, verifies the immutable source
binding before launch or resume and again on the terminal result, and stores only
exact ready output-version URIs. No workflow is created by the default dry-run
mode. A sidecar advisory lock serializes the complete load/create, launch, poll,
and manifest-write lifecycle across local processes so two launchers cannot
create a workflow for the same planned variant.

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
wrappers when cameras are replaced or deleted. The runner creates a fourth
viewer-only clone of the first exterior camera and selects that clone through
`isaac.set_active_camera`, with an `isaac.execute_script` compatibility fallback
for older hosted images. Interactive viewport navigation can therefore move the
viewer without changing any image sent to the policy. Before every sample, the
runner checks the ordered `exterior_1`, `exterior_2`, and `wrist` roles against
the upstream DROID calibration, then repeats the check after all three frames
are captured. Exterior poses are verified in world space and the wrist mount in
gripper-local space. Projection, pose, unit scale, focal length, focus distance,
apertures and offsets, f-stop, the canonical `(0.01, 1e6)` clipping range, and
the absence of custom clipping planes must all match and remain finite. Missing
or drifted calibration fails before inference rather than feeding a visually
plausible wrong view to the policy. Calibration failures include expected and
observed optics plus per-field error magnitudes in `error.json`, so a rejected
episode identifies the exact intrinsic that changed.
Explicit pre-dispatch `BRIDGE_OFFLINE` and `ISAAC_UNREACHABLE` responses are
retried with the configured readiness poll interval. Control-plane DNS and
connection failures reported before MCP bridge dispatch receive the same
bounded retry window, so an API container replacement does not discard a live
episode. Other command failures are not replayed, so ambiguous or
application-level errors still fail closed. An HTTP 502 is replayed only for
absolute `isaac.set_joint_positions` targets, where sending the identical
target again is idempotent. Read-only camera capture and simulation-state calls
also receive bounded transport retries while physics remains paused;
non-idempotent simulation steps still fail closed on transport ambiguity.
An action is appended to evidence only after `step_simulation` reports the full
requested frame count without `timed_out`; partial steps remain failure evidence,
not acknowledged actions. Before creating a fresh camera generation, the runner
deletes older evaluator-owned external roots and wrist cameras. The updated
extension releases their render products, keeping retained sessions bounded.
An object-motion, contact-penetration, contact-impulse, or grasp-support
violation is irreversible for task acceptance, so the evaluator stops at that
action and records the exact terminal reason instead of spending the remaining
action budget on an episode that can no longer pass.
Sessions launched successfully by the evaluator remain running after the
rollout. Use `--stop-session` to opt into cleanup. Attached sessions always
remain caller-owned. Cleanup always attempts both sampling-client close and an
opted-in simulator stop. A cleanup error after an otherwise successful rollout
is recorded as failure evidence and never as `result.json` success.

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
that point. Legacy `status: "succeeded"` means the evaluator completed its
requested action loop. `execution_status` states transport/execution completion,
while `task_status` independently records `passed`, `failed`, or
`not_evaluated`; a completed rollout that misses the bowl is not a task pass.
The evidence layout is:

```text
<results-dir>/
|-- config.json
|-- actions.jsonl                       # samples, accepted targets, applied actions
|-- task-states.jsonl                   # opt-in geometry/contact acceptance verdicts
|-- result.json                         # completed execution; task status is separate
|-- error.json                          # execution failure only
|-- rollout.mp4                         # post-action exterior-camera rollout
|-- runtime.json                        # measured physics dt and control cadence
|-- evidence-manifest.json              # written last; hashes every other file
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

`evidence-manifest.json` is written only after the terminal record and contains
the byte length and SHA-256 of every other file. Its aggregate identity also
binds normalized terminal semantics plus separate artifact-producer and
manifest-writer revisions. Verification rejects changed, missing, symlinked, or
unlisted files, status contradictions, and provenance or terminal-record edits.
Live runs capture their producer automatically. To assign an evidence identity
to a completed directory from an older runner, finalize it only after all
recovery work is complete and provide the revision that actually produced it:

```bash
python finalize_hosted_evidence.py \
  runs/hosted-production/<RUN_DIRECTORY> \
  --artifact-revision <PRODUCING_SIM_EVALS_GIT_SHA>
```

Omitting `--artifact-revision` marks historical producer provenance as unknown;
such evidence remains integrity-checkable but is ineligible as a replay source.
Running the finalizer again after any mutation creates a new evidence identity.

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
