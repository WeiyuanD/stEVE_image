# State RNN-SAC fixed-seed evaluation

From `/home/mtgroup/WeiyuanD/stEVE_training`:

```bash
python -m training_scripts.eval_seeds \
  --config training_scripts/configs/rnn_sac/eval_seeds.yaml \
  --checkpoint results/rnn_sac_state_150_450/best_ckpt/checkpoint6650107.everl \
  --output results/rnn_sac_seed_eval_typeI_001
```

For all checkpoints, replace `--checkpoint FILE` with:

```text
--checkpoints-dir results/rnn_sac_state_150_450/best_ckpt
```

They run newest training-step filename first. For Type II, set `arch_type: II`
in the YAML or add `--arch-type II`. Use a new output directory for each run.
From another working directory, invoke the script by absolute path and use
absolute checkpoint/config/output paths. Only load trusted checkpoints.

The config explicitly contains the existing 100 unique fixed seeds (original
98 plus 190 and 180), constructor_seed=0 and simulation_seed=0, matching the
TD-MPC2 seed evaluator defaults. Architecture/weights are reconstructed from
checkpoint metadata via the existing eval_suite loader. This entry point accepts
only the current 16-value state/LSTM policies and two physical action components.
It creates ArchVarietyV2(arch_type=...) and StEveEnv(obs='state', mode='eval').
The same ordered state observations and single normalized-to-physical action
mapping used in training feed the policy. No replay, training, W&B or optimization
is started. Default limit is 450 steps; vessel-end truncation is disabled by eval
mode, while the physical vessel-end movement constraint remains active.

Each seed resets anatomy and target. A fresh environment is created for every
checkpoint; geometry hashes and target coordinates are checked across checkpoints.
Noise augmentation is not used. Matching seeds across algorithms requires matching
the arch type, constructor/simulation seeds, environment code and instrument/target
settings as well. Seeds are not the archived 200-case suite and do not guarantee
100 unique geometries or balanced target-branch counts.

Like eval_suite, the head/body LSTM memory is explicitly reset for each episode.
The inherited deterministic SACPlayOnly action method (including its two forward
calls) is preserved; no training algorithm code is altered. Navigation time is
wall-clock seconds after reset and before output writing, including inference
and stepping. It is not simulated time and depends on machine load.

Per checkpoint: episodes.csv/jsonl, evaluation.json, summary.json,
summary_overall.csv, summary_by_anatomy.csv, summary_by_target.csv.
At batch level: config.yaml, batch.json, case_identities.json and
checkpoints_summary.csv. Success rate uses percent; mean navigation time includes
successes only; mean path ratio includes all unsuccessful episodes. Empty groups
have null JSON values / blank CSV cells, with sample counts alongside means.
Groups reflect the anatomy IDs and target branches actually sampled, not a fixed
20-anatomy/5-branch grid. Errors stop the batch, mark metadata failed and retain
completed episode rows and previous checkpoint summaries; partial runs are not
reported as complete 100-case results. This does not resume partially completed
evaluations automatically.

Implementation was inspected without tests, model execution or simulator runs.
