# State and image RNN-SAC

Reusable image encoding lives in
`eve_rl/eve_rl/network/component/imagelstm.py`, exported as
`eve_rl.network.component.ImageLSTM`. Task setup lives in
`training_scripts/util/image_env.py` and `image_agent.py`; the entry point is
`training_scripts/RNNSAC_train.py`. There is no separate `rnn_sac` package or
custom SAC subclass.

The implementation was reviewed by reading the source. No tests, Python imports,
simulation, or training were run for this change, as requested.

## Commands for later use

From the `stEVE_training` root, use the Python environment containing the local
editable `eve`, `eve_bench`, and `eve_rl` packages, SOFA, PyTorch, Gymnasium,
NumPy, Pillow, and PyYAML. The local eve_rl installation must include ImageLSTM.

```bash
python -m pip install --no-deps -e ./eve_rl
python -m training_scripts.RNNSAC_train --config training_scripts/configs/rnn_sac/state.yaml --output results/rnn_sac_state
python -m training_scripts.RNNSAC_train --config training_scripts/configs/rnn_sac/image_single.yaml --output results/rnn_sac_image_single
python -m training_scripts.RNNSAC_train --config training_scripts/configs/rnn_sac/image_stacked.yaml --output results/rnn_sac_image_stacked
```

Choose one command per experiment. Output directories must be new. All presets
inherit `base.yaml`; nested overrides merge recursively. Relative `extends`
paths resolve against the containing config. Image presets default to `cuda`;
the state preset defaults to `cpu`. Keep the workspace root importable for environment
reconstruction from checkpoints.

If importing `Runner` reports `eve_rl (unknown location)`, Python is finding
the outer repository directory rather than the installed package. The
RNNSAC_train entry point now puts this workspace's `eve_rl` source directory
first on Python's import path before importing agent helpers, including in
spawned workers. The editable-install command above remains useful for other
entry points and standalone imports; `--no-deps` keeps existing dependencies.

The image presets use one Single agent and full-episode VanillaEpisode replay.
The state preset uses Synchron with two CPU workers, a CPU trainer, and
VanillaEpisodeShared replay, matching the original script's defaults. All use
the existing Runner. Outputs include the resolved configuration, environment
constructor configurations, training log, results CSV, and `.everl` checkpoints.
Replay contents are not saved; see the checkpoint continuation instructions below.

## Data flow

| Preset | Per-step observation | Head |
|---|---|---|
| state | Existing tracking memory, target, last action | Existing LSTM |
| image_single | One grayscale frame and previous command | CNN + fusion + LSTM |
| image_stacked | Two grayscale frames and previous command | CNN + fusion + LSTM |

State mode preserves StEveEnv's original state observation (16 values for the
current single-device task), including its own two-step tracking memory. No
additional state stacking is applied. Image mode uses StEveEnv's guidewire image
with vessels and target marker already drawn; no target coordinates or tracking
coordinates are supplied to the policy.

Image observations have shape `[F,96,96]`, oldest frame first. At reset, the
initial frame fills the stack and action history is zero. The image adapter
converts [0,255] pixels to uint8, as the TDMPC2 wrapper does. It returns `image`
before `last_action`; the declared dictionary space preserves the same order.

```text
observation: image [F,96,96], optional command history [A]
flattening:  [F*96*96 + A]
replay:      [B,T+1,F*96*96 + A], padded full episodes
CNN:         [B*(T+1),F,96,96] -> [B,T+1,feature_dim]
fusion:      append action values or add a learned action embedding
LSTM:        [B,T+1,hidden_size]
actor:       recurrent features -> policy MLP
critic:      recurrent features + current candidate action -> Q MLP
```

The existing SAC update selects T current observations and next-state targets
from T+1 observations. Training forwards reconstruct history from zero and do not
use stored rollout state. Online forwards maintain state between calls.

The CNN uses 5x5 convolutions, max pooling, channel RMS normalization, SiLU, and
a feature projection, following the basic small TDMPC2 encoder design. Pixels
are normalized once as `x / 255 - 0.5`. No random-shift augmentation or SimNorm
is included. `concat` imitates the state model's input concatenation; `sum`
adds a learned action embedding to image features. Sum fusion is analogous to,
but not an exact copy of, the TDMPC2 action encoder.

Set `include_previous_action: false` for image-only inputs. `frame_stack` and
`action_stack` independently control positive history lengths in image mode.
The default is one previous command and concatenation for both image presets.

eve_rl maps normalized commands to physical actions exactly once. The adapter
inverts that mapping solely to record the preceding command. It does not clip or
rescale simulator inputs. This is command history, not applied simulator motion.
Inherited exploration noise can exceed [-1,1], so the history space allows those
values. State mode retains its original LastAction semantics.

The actor and both critics share one observation head, exactly as in the existing
state helper. The Q1 optimizer owns the entire Q1 network, including CNN/LSTM;
the other optimizers own only their MLP bodies. Thus the image head is updated
through Q1 loss. Target critics are copied/updated by existing SAC. Constructor
attributes on ImageLSTM support the existing config serializer and shared-object
reconstruction without changing checkpoint format.

## Inherited limitations

This change adds a component and task setup; SAC, replay, network reset, and
evaluation logic are retained. In the inspected eve_rl version:

- GaussianPolicy/QNetwork reset does not call the head's reset, so rollout memory
  can persist across episodes. ImageLSTM exposes reset but these callers do not
  propagate it, just as with the existing state LSTM.
- Deterministic evaluation processes each observation through forward_play twice.
- Replay stores terminal flags, not truncations; existing bootstrap and padded
  loss behavior are retained.

These are known correctness limitations of the inherited recurrent execution.
The new data flow is not a claim that the complete recurrent algorithm is
validated. Addressing them is separate from the agreed image-support scope.

Train and evaluation environments are separate instances. Image presets use
the same stopping mode for both; the state preset uses train/eval modes as in
the original script. Replay capacity counts episodes; mixing image/action arrays in the existing
flattening promotes images to floating point. Full-episode CNN batches can use
substantial memory. Image defaults are 200 replay episodes and batch size 4; GPU memory
has not been measured.

When non-null, the seed initializes Python, NumPy and PyTorch; the state preset
leaves it null, matching the original script's unseeded training setup.
Evaluation uses explicit environment seeds. Deterministic simulation and held-out geometry are not
guaranteed. Environment configs reconstruct the current default ArchVarietyV2;
they do not snapshot source code, active geometry, or simulator state.

## Original-compatible state preset

`state.yaml` matches the defaults in `training_scripts/BasicWireNav_train.py`,
with the requested exceptions: the shared StEveEnv creates ArchVarietyV2 and
training has a 150-step limit and evaluation has a 450-step limit, matching
the original ArchVariety scripts. Evaluation now runs every
50,000 exploration steps over 100 seeded anatomy–target cases (the original
98 seeds plus 190 and 180). Distinct anatomy geometry is not explicitly enforced.

| Setting | State preset |
|---|---|
| Agent / replay | Synchron / VanillaEpisodeShared |
| Workers / devices | 2 workers, CPU workers and trainer |
| Simulator isolation | SimulationMP, as in original BenchEnv |
| Recurrent / MLP | 1 layer of 700 / [400,400,400] |
| Learning rate | 0.0003217978434614328 |
| Linear LR schedule | End factor 0.15 over 6,000,000 updates |
| Gamma / reward scaling | 0.99 / 1 |
| Replay / batch | 10,000 episodes / 32 episodes |
| Heatup / exploration steps | 500,000 / 20,000,000 |
| Exploration episodes between updates | 100 |
| Update ratio | 1/20 |
| Train / evaluation episode limits | 150 / 450 steps |
| Evaluation interval | 50,000 exploration steps (requested override) |
| Evaluation seeds | 100 seeds: original 98 plus 190 and 180 |
| Heatup physical action bounds | [-10,-1] to [25,3.14] |
| Action repeat / worker timeout | 1 / 180 seconds |
| Evaluation / metrics | Deterministic / all environment info fields |

Training stops on target success or max-step/vessel-end/simulation-error
conditions; evaluation stops on target success or max steps. `env.eval_mode`
selects this independently of `env.mode`. Custom heatup bounds are physical
units; null bounds in the image presets use the environment's full action range.
The network, optimizer ownership, and inherited recurrent limitations above
remain the same. Output filenames and configuration handling follow the new
entry point, not the old results-directory helper.

The existing eve_rl checkpoint writer records the evaluation environment config
in both environment entries. With different train/eval modes, pass explicitly
constructed training and evaluation environments when restoring an agent. The
entry point also saves the correct separate environment configs alongside the
checkpoint directory; the resume CLI rebuilds environments from the supplied YAML.

## W&B and local logging

All three presets inherit `logging` from `base.yaml`. In your training Python
 environment, install the optional logging dependencies and authenticate once:

```bash
python -m pip install wandb imageio imageio-ffmpeg
wandb login
python -m training_scripts.RNNSAC_train \
  --config training_scripts/configs/rnn_sac/state.yaml \
  --output results/rnn_sac_state_wandb_001
```

Choose a new output directory for each run. The W&B project defaults to
`rnn-sac`; set `logging.entity` to your username/team if needed. Run names
default to the output directory name. `logging.mode: offline` records a run
locally for later synchronization. `logging.enable_wandb: false` disables W&B
while retaining local CSVs and videos; each has its own switch. No W&B logging
is performed until you launch training.

- `train.csv` / `train/*`: batch mean episode reward, length and success rate;
  exploration, episode, update and heatup counters; elapsed time, average
  exploration throughput, entropy coefficient and learning rates.
- `loss.csv` / `loss/*`: mean Q1, Q2 and policy loss over updates in each batch.
  Batches without updates do not emit losses.
- `eval.csv` / `eval/*`: mean reward, length, success rate, best success rate,
  failed-episode path ratio, and successful-episode wall-clock navigation time.
  Empty subsets report zero, matching TD-MPC2. Navigation time excludes reset
  and includes inference, simulator steps and frame capture; parallel workers
  measure their own episodes. It is affected by machine load.
- `eval_episodes.csv`: one row per returned evaluation episode, with evaluation
  step, seed, success, reward, length, path ratio and wall-clock navigation time,
  including failed episodes. The existing `results.csv` remains available and
  keeps its original all-episode mean path ratio.
- `videos/`: MP4 files for fixed evaluation seeds, also uploaded to W&B.
  `video_eval_seeds: null` selects the first two configured seeds: `[1, 2]` for
  state and `[10001, 10002]` for the image presets. Specify a subset or `[]` to
  override. Videos use the environment's image frequency. State videos use a
  separate fluoroscopy observation; image videos use the current unstacked
  frame. Neither enters replay or changes the policy observation.
- `logging.save_agent: true` uploads each improved best checkpoint as a W&B
  model artifact; local checkpoint saving retains the existing behavior.

Plots use exploration steps, excluding heatup, as their horizontal axis.
Logging occurs after each exploration/update batch and each evaluation, so
there are no scalar training points during the initial heatup. Workers send
only episode metadata/selected video frames; W&B and video encoding run in the
parent. Training/evaluation scheduling, recurrent models and SAC updates retain
their existing behavior. The metrics `eval/navigation_time_simulated`,
`eval/success_count`, and `eval/failure_count` are not emitted.

### Live terminal progress

The training entry point enables a flushed terminal line after every completed
heatup and training episode, directly from the collecting worker:

```text
 heatup  E: 1            I: 200          R: -0.8         S: 0.0          T: 0:00:38
 train   E: 338          I: 42,566       R: 0.6          S: 1.0          T: 2:25:46
```

`E` is the episode ID assigned when an episode starts; parallel workers may
finish IDs out of order. `I` counts completed environment steps across workers
within the displayed phase (heatup and training have separate counters). `R`
is that episode's reward, `S` its final success flag, and `T` wall-clock time
since agent startup, including heatup. Output is immediate on episode completion,
without waiting for a training batch or W&B upload. These terminal lines use
stdout; redirect or tee stdout if you also want to save them. Existing running
Python processes must be restarted to load code changes; this entry point requires `--resume` to continue from a checkpoint.

## Continue from a checkpoint

```bash
python -m training_scripts.RNNSAC_train \
  --config results/rnn_sac_state_001/config.yaml \
  --resume results/rnn_sac_state_001/checkpoints/checkpoint60000.everl \
  --output results/rnn_sac_state_001_resumed
```

Use a new output directory. This starts a new W&B run, with restored exploration
and update counters. The supplied config must match the saved environment,
network and learning settings; worker count, logging and training schedule may
be changed. `training_steps` remains the total exploration-step target, not an
additional budget. Only load trusted checkpoints.

Before workers start, networks (including target critics and entropy coefficient),
Adam states and LR schedulers are restored. Shared counters are restored before
any rollout commands. Replay, simulator state and RNG state were not saved.
To rebuild replay, continuation collects `batch_size + 1` random-action episodes
(default 33 for state), with no updates; these add to heatup counters only.
Use `--resume-refill-episodes 200` for a larger initial refill. This replaces the
full initial heatup and is not an exact replay of the interrupted trajectory.
The minimum refill provides enough episodes to sample; it does not restore the
previous buffer's diversity. Original heatup counters are preserved and increased
by this refill. Wall-clock measurements restart for the new run.

The first resumed evaluation is scheduled one configured interval after the
restored exploration count. Episode/batch boundaries can overshoot that threshold.
After every evaluation, `checkpoint<actual_exploration_steps>.everl` is saved
locally, before media logging. `best_checkpoint.everl` is updated only on a strict
improvement. New checkpoints include historical best metadata; for older files,
resume also consults the adjacent best checkpoint if it is no newer than the
selected checkpoint. Until a new best is achieved, the inherited best file remains
in the original output folder. `resume.yaml` records the source checkpoint and
refill settings. Checkpoints still do not include replay data.

The state preset uses `env.n_max_steps: 150` and `env.eval_n_max_steps: 450`.
When `eval_n_max_steps` is omitted (including older saved configs and the image
presets), evaluation inherits `n_max_steps`. Existing running processes retain
their original limits. Resuming with an old saved config retains its old limits;
resume validation rejects switching environment settings mid-run.

## Optional grayscale image augmentation

Default `env.image_augmentation: null` preserves the current clean pixels.
For an image preset, configure:

```yaml
env:
  image_augmentation:
    noise_std: 8.0
    brightness: -15.0
    contrast: 0.9
    seed: 42
```

These are grayscale intensity changes, not RGB tinting. The calculation is
`clip(round((clean - 127.5) * contrast + 127.5 + brightness + noise), 0, 255)`.
Noise is independent Gaussian pixel noise with standard deviation `noise_std`
in 8-bit intensity units, freshly sampled each frame. Brightness and contrast
are fixed parameters. Identity values are 0, 0 and 1 respectively. The separate
noise seed reproduces a sequence given the same frame/reset call sequence; it
is not reset by the anatomy seed. With multiprocessing, copied RNG states can
produce matching noise sequences across workers. This is not an X-ray physics
model. The option applies to both image training and image evaluation. State
observations and their diagnostic videos retain their existing behavior.

Flow: Pillow clean guidewire → ShowVessels vessel/target composition → optional
augmentation → image observation → RNN-SAC frame stacking. The clean image is
retained as `ShowVessels.clean_obs` for comparisons and never passed to the policy
when augmentation is enabled. Default TD-MPC2/Dreamer factory calls remain clean;
their factories must explicitly pass this option to StEveEnv to enable it.

Save a clean and augmented pair from exactly the same anatomy/target/reset:

```bash
python eve/examples/function_check_fluoro.py \
  --seed 1 \
  --image-config eve/examples/fluoro_augmentation.yaml \
  --save-image-pair results/fluoro_pair_001
```

This constructs the shared StEveEnv image environment, resets once, saves
`clean.png`, `augmented.png` and `settings.json`, closes SOFA and exits without
a GUI. Use a new output directory. Omit `--image-config` to save two identical
clean images. The supplied YAML is an opt-in example with visible changes;
it does not alter training defaults. The script also accepts a resolved run
`config.yaml` containing `env.image_augmentation` (not an inherited preset).
Without `--save-image-pair`, `--image-config` applies the effect in the normal
interactive and evaluation-seed viewers.

The image example also accepts top-level `anatomy_seed: 1` in
`eve/examples/fluoro_augmentation.yaml`. It defaults to `null`; `--seed` overrides
it. This uses the existing environment reset seed, so both anatomy and target
selection are reproducible, independently of `image_augmentation.seed`. In the
interactive multi-episode example, reset seeds advance as `anatomy_seed + episode`.
It does not modify training anatomy sampling.
