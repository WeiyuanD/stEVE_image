# Archived 200-case evaluation

From the repository root, in the existing SOFA/PyTorch environment:

```bash
python -m training_scripts.eval_suite \
  --suite /home/mtgroup/WeiyuanD/11_dataset/evaluation_sets/type_I/200_cases_20_anatomies_10_targets \
  --checkpoints-dir /home/mtgroup/WeiyuanD/stEVE_training/results/rnn_sac_state_150_450/best_ckpt \
  --training-config results/rnn_sac_state_150_450/config.yaml \
  --max-steps 450 \
  --device cpu \
  --output results/rnn_sac_typeI_suite_eval_001
```

Use a new output directory. All matching `*.everl` files directly in the folder
are evaluated in descending natural filename order (highest step first for
`checkpoint<step>.everl`), one checkpoint and one case at a time.
Optional `--pattern 'checkpoint*.everl'` narrows selection. Only trusted local
checkpoints should be loaded. This script supports the current state-based
RNN-SAC checkpoints; image checkpoints are rejected.

The suite must contain 200 cases, 20 distinct anatomy IDs and 5 target branches.
All saved cases are loaded through CaseSuite, preserving initial-image and
mechanical-state verification. State features are generated afterward using
make_state_observation (tracking memory, target, last action), in the original
ordered dictionary layout. Checkpoint network construction and weights come
from checkpoint metadata. No replay buffer, heatup, optimization or training
workers are created. The supplied training config is checked for state mode and
copied for provenance; network definitions are authoritative in each checkpoint.

Archived truncation rules are replaced with MaxSteps(450), not combined with it.
VesselEnd cannot terminate an episode; physical movement constraints at vessel
ends remain unchanged. Target-success termination remains active. Simulator
exceptions mark the checkpoint evaluation failed rather than manufacturing a
valid failure metric. Failed checkpoints retain completed episode rows and the
batch proceeds to the next checkpoint. Only fully completed evaluations get
aggregate summaries. An incomplete batch exits with a nonzero status.

Policy inference uses the existing deterministic SACPlayOnly.get_eval_action
(including its inherited double forward pass). The policy head/body are explicitly
reset at each case to clear LSTM history, unlike the upstream no-op policy reset.
This evaluation therefore isolates cases but is not bit-for-bit identical to
historic training-time evaluations that carried hidden state across episodes.
No training algorithm code is changed. Actions are mapped once from normalized
policy outputs to the physical bounds of the archived environment.

Outputs:

```text
batch.json
training_config.yaml
checkpoints_summary.csv
checkpoint3905686/
  evaluation.json
  episodes.csv
  episodes.jsonl
  summary.json
  summary_overall.csv
  summary_by_anatomy.csv
  summary_by_target.csv
...one directory per checkpoint
```

Each summary reports success_rate as a percentage (0–100), mean_navigation_time
in wall-clock seconds for successful episodes only, and
mean_path_ratio_unsuccessful for all unsuccessful episodes, not just truncated
ones. Empty subsets are null in JSON / blank in CSV (not misleading zero times).
Sample counts accompany the conditional means. Wall time excludes reset and
includes policy inference, environment stepping and observation updates. There
is no rendering, video encoding or W&B upload in this evaluator. Each episode
row includes identifiers, reward, steps, success, final path ratio and wall time.
Checkpoint hashes and the suite identity are recorded to identify the exact
inputs. Twenty anatomy summaries and five target-branch summaries are produced
for every completed checkpoint. Target groups use archived branch labels, not
individual target coordinates.

No tests or simulator evaluations were run when implementing this script.
