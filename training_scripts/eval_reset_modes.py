"""Standalone fixed-seed LSTM reset comparison; sys_test is not involved."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.eval_seeds import anatomy_digest, target_branch
from training_scripts.eval_suite import digest, load_policy, summarize, write_csv, write_json
import numpy as np
import torch
import yaml
from eve_bench import ArchVarietyV2
from eve_bench.steve_env import StEveEnv
from eve_rl.util import flatten_obs


def reset_memory(algo):
    algo.reset()
    algo.model.policy.head.reset()
    algo.model.policy.body.reset()


def evaluate_run(path, output, config, identities, algo, step, run_number):
    output.mkdir()
    metadata = dict(checkpoint=str(path), run=run_number, config=config,
                    status='running', completed_episodes=0)
    write_json(output / 'evaluation.json', metadata)
    policy = algo.model.policy
    env = None
    rows = []
    try:
        # Restart simulator RNG identically each run; only policy memory differs.
        intervention = ArchVarietyV2(arch_type=config['arch_type'],
                                   anatomy_seed=config['constructor_seed'],
                                   simulation_seed=config['simulation_seed'])
        env = StEveEnv(intervention=intervention, obs='state', mode='eval',
                       n_max_steps=config['max_steps'])
        low, high = env.action_space.low, env.action_space.high
        if intervention.normalize_action or low.size != 2 or not np.isfinite([low, high]).all() or not (high > low).all():
            raise ValueError('Expected finite physical action bounds, without environment normalization')
        for index, seed in enumerate(config['eval_seeds'], 1):
            obs, _ = env.reset(seed=seed)
            anatomy = anatomy_digest(intervention.vessel_tree)
            target = np.asarray(intervention.target.coordinates3d).tolist()
            case_identity = dict(anatomy_id=anatomy, target_coordinates3d=target)
            if seed in identities and identities[seed] != case_identity:
                raise RuntimeError(f'Anatomy/target changed across checkpoints for seed {seed}')
            identities[seed] = case_identity
            branch = target_branch(intervention)
            if config['reset_mode'] == 'episode':
                reset_memory(algo)
            total = 0.0
            start = perf_counter()
            for steps in range(1, config['max_steps'] + 1):
                flat, _ = flatten_obs(obs)
                if flat.size != policy.n_observations or not np.isfinite(flat).all():
                    raise ValueError('Invalid state observation')
                with torch.inference_mode():
                    action = np.asarray(algo.get_eval_action(flat), dtype=np.float32)
                if action.size != low.size or not np.isfinite(action).all():
                    raise ValueError('Invalid policy action')
                physical = (action.reshape(low.shape) + 1) / 2 * (high - low) + low
                obs, reward, terminated, truncated, info = env.step(physical)
                total += float(reward)
                if terminated or truncated:
                    break
            elapsed = perf_counter() - start
            ratio = float(info['path_ratio'])
            if not np.isfinite(ratio) or not np.isfinite(total):
                raise ValueError('Nonfinite episode metrics')
            row = dict(run=run_number, reset_mode=config['reset_mode'], checkpoint_step=step, case_id=f'seed_{seed:06d}', environment_seed=seed,
                       arch_type=config['arch_type'], anatomy_id=anatomy, target_branch=branch,
                       target_coordinates3d=json.dumps(target), success=bool(info['success']),
                       steps=steps, **{'return': total}, final_path_ratio=ratio,
                       wall_time_s=elapsed, terminated=bool(terminated), truncated=bool(truncated))
            rows.append(row)
            write_csv(output / 'episodes.csv', rows)
            with (output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            metadata['completed_episodes'] = index
            write_json(output / 'evaluation.json', metadata)
            print(f'{path.name} | run={run_number} | {index}/100 | seed={seed} | {branch} | '
                  f'success={int(row["success"])} steps={steps} path_ratio={ratio:.4f}', flush=True)
        summary = summarize(output, rows)
        metadata['status'] = 'complete'
        return summary
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'evaluation.json', metadata)
        if env is not None:
            env.close()


def summarize_runs(output, summaries, expected_runs):
    metrics = ('success_rate', 'mean_navigation_time', 'mean_path_ratio_unsuccessful')
    for section, filename, identity_key in (
            ('overall', 'summary_overall', 'group_id'),
            ('by_branch', 'summary_by_branch', 'target_branch'),
            ('by_anatomy', 'summary_by_anatomy', 'anatomy_id')):
        tables = [({'overall': s['overall']} if section == 'overall' else s[section])
                  for s in summaries]
        identities = sorted(set().union(*(set(t) for t in tables)))
        consistent = all(set(t) == set(identities) for t in tables)
        ready = len(summaries) == expected_runs and consistent
        rows = []
        for identity in identities:
            for metric in metrics:
                values = [t[identity][metric] for t in tables
                          if identity in t and t[identity][metric] is not None]
                rows.append(dict(**{identity_key: identity}, metric=metric,
                    completed_runs=len(summaries), expected_runs=expected_runs,
                    valid_runs=len(values), status='complete' if ready else 'incomplete',
                    mean=statistics.mean(values) if ready and values else None,
                    std=statistics.stdev(values) if ready and len(values) > 1 else None))
        write_csv(output / f'{filename}.csv', rows)
        write_json(output / f'{filename}.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/rnn_sac/reset_episode.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; no resume')
    parser.add_argument('--arch-type', choices=['I', 'II'])
    args = parser.parse_args()
    options = yaml.safe_load(args.config.read_text())
    if not isinstance(options, dict) or set(options) - {'base_config', 'reset_mode', 'runs'}:
        parser.error('Reset config accepts only base_config, reset_mode, runs')
    base = args.config.resolve().parent / options.get('base_config', 'eval_seeds.yaml')
    config = yaml.safe_load(base.read_text())
    expected = {'arch_type', 'constructor_seed', 'simulation_seed', 'max_steps', 'device', 'threads', 'eval_seeds'}
    if not isinstance(config, dict) or set(config) != expected:
        parser.error('base_config must be a standard eval_seeds YAML')
    config.update(reset_mode=options.get('reset_mode', 'episode'), runs=options.get('runs', 10))
    if args.arch_type:
        config['arch_type'] = args.arch_type
    if config['reset_mode'] not in ('episode', 'run', 'never'):
        parser.error('reset_mode must be episode, run, or never')
    if config['arch_type'] not in ('I', 'II'):
        parser.error('arch_type must be I or II')
    for key in ('constructor_seed', 'simulation_seed', 'max_steps', 'threads', 'runs'):
        minimum = 1 if key in ('max_steps', 'threads', 'runs') else 0
        if type(config[key]) is not int or config[key] < minimum:
            parser.error(f'Invalid {key}')
    seeds = config['eval_seeds']
    if (not isinstance(seeds, list) or len(seeds) != 100 or
            any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 100):
        parser.error('Expected exactly 100 unique nonnegative episode seeds')
    path = args.checkpoint.expanduser().resolve()
    if not path.is_file():
        parser.error(f'Checkpoint not found: {path}')
    torch.set_num_threads(config['threads'])
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(status='running', checkpoint=str(path), checkpoint_sha256=digest(path),
                    config=config, completed_runs=0,
                    initial_memory='zero once before the first episode',
                    environment_reset='every episode; fresh environment and simulator RNG each run',
                    policy_action='existing deterministic get_eval_action (two recurrent forwards)',
                    resume_supported=False)
    write_json(output / 'experiment.json', metadata)
    algo = None
    summaries, run_rows, identities = [], [], {}
    try:
        algo, step = load_policy(path, config['device'])
        policy = algo.model.policy
        if policy.n_observations != 16 or policy.n_actions != 2 or type(policy.head).__name__ != 'LSTM':
            raise ValueError('Expected a state policy with 16 inputs, 2 actions and an LSTM head')
        # Even never mode starts from a defined zero state; it never resets again.
        reset_memory(algo)
        for run_number in range(1, config['runs'] + 1):
            if config['reset_mode'] == 'run':
                reset_memory(algo)
            summary = evaluate_run(path, output / f'run_{run_number:02d}', config,
                                   identities, algo, step, run_number)
            summaries.append(summary)
            run_rows.append(dict(run=run_number, **summary['overall']))
            write_csv(output / 'runs.csv', run_rows)
            summarize_runs(output, summaries, config['runs'])
            metadata['completed_runs'] = run_number
            write_json(output / 'experiment.json', metadata)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'case_identities.json', identities)
        write_json(output / 'experiment.json', metadata)
        if algo is not None:
            algo.close()


if __name__ == '__main__':
    main()
