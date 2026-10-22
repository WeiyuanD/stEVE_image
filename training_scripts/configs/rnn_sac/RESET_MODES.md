# Fixed-seed RNN-SAC recurrent-memory comparison

Run `training_scripts.eval_reset_modes` directly, independently of sys_test.
Neither the original eval_seeds/eval_suite evaluators nor sys_test are changed.

| Config | LSTM memory boundaries |
| --- | --- |
| reset_episode.yaml (default) | Clear before every episode |
| reset_run.yaml | Clear at the start of each 100-episode run |
| reset_never.yaml | Initialize to zero once; retain across all 1,000 episodes |

All modes load the checkpoint once and run ten repetitions in the same process.
Only policy memory persistence differs. The environment resets for every episode,
and a fresh environment with the same simulator seed is constructed each run.
Thus `never` does not mean disabling environment resets. Simulator initialization
uses the base config's simulation_seed and advances within each run. The existing
deterministic action method, including its two recurrent forwards, is unchanged.
These experiments do not reproduce historical training worker/memory/RNG histories.

The configs inherit the exact 100 seeds, 450-step limit, CPU/device settings,
constructor seed and simulation seed from eval_seeds.yaml. Environment eval mode
has no VesselEnd truncation; target success still terminates the episode. Use
--arch-type I or --arch-type II. Keep seed order fixed because memory persistence
makes order part of the experiment.

From stEVE_training, in the environment containing stEVE/SOFA/PyTorch:

```bash
python -m training_scripts.eval_reset_modes \
  --checkpoint results/rnn_sac_state_150_450/best_ckpt/checkpoint4900075.everl \
  --arch-type I \
  --output results/reset_comparison/4900075/typeI/episode
```

Omitting --config selects reset_episode.yaml. For the other modes, supply:

```text
--config training_scripts/configs/rnn_sac/reset_run.yaml
--config training_scripts/configs/rnn_sac/reset_never.yaml
```

Use distinct output directories for each mode/anatomy/experiment. Each invocation
runs all ten repetitions itself; do not wrap it in sys_test's ten-run launcher.
That would create 100 repetitions and would break memory continuity between
launcher processes.

Outputs include resolved config, checkpoint hash, completion status, case identities,
per-run episode CSV/JSONL and overall/branch/anatomy summaries. Root runs.csv holds
one row per run; summary_overall.csv, summary_by_branch.csv and
summary_by_anatomy.csv contain equal-run-weight mean/sample std (ddof=1). Empty
conditional metrics stay null. Statistics remain incomplete until all ten runs
finish. An exception aborts the sequence and leaves failure metadata.

There is intentionally no resume: restarting halfway would discard the LSTM history
required by never mode. Restart an interrupted comparison using a new output path.
Runs in never mode are sequential blocks of one continuous memory trajectory,
not independent evaluation replicates; interpret their std accordingly.

No tests, checkpoint loads, or simulator evaluations were run during implementation.
