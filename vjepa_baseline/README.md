# Frozen V-JEPA2 behavior cloning for stEVE

This package trains a goal-conditioned imitation policy from the existing
`demo_datasets/*/episodes/*.h5` files. It uses **V-JEPA2 ViT-L**, not V-JEPA2-AC.
It does not modify stEVE, collect data, or train the video encoder.

**Status: implemented but not executed or tested.** The commands below are
instructions for a later run, not a record of successful validation.

## Components

- `data.py`: indexes valid causal windows, repeats the initial image for short
  histories, reads images lazily from HDF5, and creates whole-group splits.
- `encoder.py`: constructs the ViT-L encoder directly, loads `target_encoder`
  from a local official checkpoint, freezes it, and shares preprocessing between
  extraction and online evaluation. It does not instantiate the unused predictor
  or use the checkout's localhost checkpoint URL.
- `extract.py`: builds train/validation/test feature caches, including provenance,
  episode/step indices, targets, previous actions, and teacher actions.
- `train.py`: fits a 1028 → 256 → 256 → 2 ReLU MLP with tanh outputs using MSE.
  Selects `best.pt` on validation MSE; never opens the test cache.
- `evaluate.py`: runs the learned policy in `ArchVarietyV2` through `StEveEnv`'s
  image observations. It reports per-episode results and aggregate metrics.

## Data contract and alignment

One episode contains N normalized actions and N+1 grayscale uint8 observations
of shape 96×96×1. Action t follows observation t. An eight-frame input ends at t,
never t+1; early windows repeat observation zero. Windows cannot cross episodes.
The loader filters invalid observations/transitions/applied actions and invalid
target pixels. It also checks timestamps and requires a fixed single-step control
interval equal to `1 / image_frequency`.

Targets use the collector's renderer `(x,y)` pixel convention and are normalized
as `pixel / 48 - 1`. Previous action is the preceding normalized teacher command
during training and the preceding policy command during evaluation; reset uses
zeros. Simulator tracking and tip coordinates are not policy inputs.

Images are repeated into three channels and processed by the official 256-pixel
evaluation transform (resize, center crop, tensor conversion, normalization).
All encoder output tokens are mean-pooled into 1024 features. Target coordinates
remain in the original image coordinate system as a separate global input; they
are not interpreted as coordinates in the encoder's cropped patch grid.

Offline and live intensities must both be in [0,255]. The adapter matches the
collector's uint8 conversion; it does not automatically rescale [0,1] images.

## Installation assumptions

Activate the environment containing your local eve/eve_bench installations,
PyTorch, torchvision, timm, einops, h5py, and NumPy. No Dreamer or JAX import is
required. Keep the V-JEPA2 checkout available for its model/preprocessing code.
All commands run from the stEVE_training root so this package is importable.

```bash
conda activate sofa_vjepa
cd ~/WeiyuanD/stEVE_training
```

Download the official ViT-L checkpoint once if you do not already have it:

```bash
mkdir -p checkpoints
curl --fail --location --output checkpoints/vitl.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
```

Use only a trusted official checkpoint. The loader uses PyTorch's restricted
`weights_only=True` loading and rejects architecture mismatches. GPU operation
is the default; choose `--device cpu` explicitly if needed. No GPU memory or
latency claim has been measured for this implementation.

## 1. Extract features

Begin with the 10-episode dataset when you are ready to check the pipeline:

```bash
python -m vjepa_baseline.extract \
  --dataset ~/WeiyuanD/demo_datasets/00_10eps_200max_mix_state_ckpt7655456 \
  --repo ~/WeiyuanD/phd_project/code/03_v_jepa2/vjepa2 \
  --checkpoint checkpoints/vitl.pt \
  --output results/vjepa_small_cache \
  --window 8 --batch-size 1 --device cuda
```

For the main baseline:

```bash
python -m vjepa_baseline.extract \
  --dataset ~/WeiyuanD/demo_datasets/03_5000eps_200max_suc_state_ckpt_7655456 \
  --repo ~/WeiyuanD/phd_project/code/03_v_jepa2/vjepa2 \
  --checkpoint checkpoints/vitl.pt \
  --output results/vjepa_cache \
  --window 8 --batch-size 1 --device cuda
```

Default splits allocate roughly 80/10/10 percent of **geometry groups**, so
episode counts need not match those percentages. At least three distinct groups
are required. `--group-by seed` or `--group-by episode` is an explicit alternative;
episode grouping does not establish unseen-geometry generalization. Persisted
`splits.json` lists exact assignments; reuse it through `--split-file` for encoder
comparisons. Do not mix dataset folders without checking duplicates first.

The output directory must be new. Incomplete extraction files end in
`.partial.h5`; training refuses incomplete caches. There is no extraction resume
support yet. Full extraction encodes every overlapping window independently and
can be expensive. Increase batch size only after measuring available memory.

Each cache contains features, four auxiliary values, action labels, episode
indices, step indices, and episode paths. Provenance includes checkpoint SHA256,
V-JEPA2 Git commit, split-manifest SHA256, preprocessing, window length, action
bounds, and control frequency. Keep the external checkout unchanged: its Git
commit identifier does not detect uncommitted source edits.

## 2. Train behavior cloning

```bash
python -m vjepa_baseline.train \
  --cache results/vjepa_cache \
  --output results/vjepa_bc_seed0 \
  --epochs 50 --batch-size 256 --lr 0.0003 --seed 0 --device cuda
```

Repeat with seeds 1 and 2 and different output directories. Training logs overall
and per-action train/validation MSE in `metrics.jsonl`, and saves the best
validation checkpoint to `best.pt`. Frames are not loaded during policy training.
The caches are read lazily, so the complete feature dataset need not fit in RAM.
Training seeds are recorded, but bitwise GPU reproducibility is not guaranteed.

## 3. Evaluate in stEVE

Example syntax (replace these example seeds with a verified held-out seed list):

```bash
python -m vjepa_baseline.evaluate \
  --policy results/vjepa_bc_seed0/best.pt \
  --repo ~/WeiyuanD/phd_project/code/03_v_jepa2/vjepa2 \
  --checkpoint checkpoints/vitl.pt \
  --output results/vjepa_eval_seed0 \
  --seeds 10001 10002 10003 \
  --max-steps 200 --device cuda
```

This launches a live simulation. The adapter deliberately uses `mode='train'`
to preserve the collector's vessel-end and simulation-error truncations; this
does not update the policy. Normalized commands are mapped once into the actual
environment bounds using the same affine mapping as the collector. There is no
additional action repeat. Invalid live observations fail visibly instead of
being silently replaced.

The adapter uses the collector's private fluoroscopy pixel transform, so a future
stEVE API change may require updating that call. It constructs a fresh
ArchVarietyV2 rather than restoring arbitrary dataset environment configurations.
It checks bounds and image frequency but does not establish complete environment
equivalence. Keep the same stEVE implementation and relevant simulator settings.
Explicit evaluation seeds alone do not prove held-out geometry, especially when
the collection metadata has null seeds: inspect the recorded geometry identifiers
before making generalization claims.

Outputs: `config.json`, `episodes.jsonl`, and `summary.json`. Metrics include
success, return, successful-episode length, vessel-end/simulation-error/timeout
rates, and wall-clock policy inference time including preprocessing and encoding.
Failure flags can overlap. Model/adapter errors are logged and raised, not counted
as ordinary simulation outcomes. Timing includes first-call warmup.

## Scope

This is a first frozen-feature BC baseline. No AC world model, encoder fine-tuning,
CNN comparison, offline test-score command, checkpoint resume, or multi-worker
GPU inference service is included. Closed-loop success is the main outcome;
validation action MSE alone does not demonstrate a successful controller.
