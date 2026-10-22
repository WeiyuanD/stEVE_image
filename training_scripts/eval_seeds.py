"""Evaluate state RNN-SAC checkpoints on 100 seeded ArchVarietyV2 cases."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

# Support both python -m training_scripts.eval_seeds and an absolute script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.eval_suite import digest, load_policy, summarize, write_csv, write_json

import numpy as np
import torch
import yaml
from eve_bench import ArchVarietyV2
from eve_bench.steve_env import StEveEnv
from eve_rl.util import flatten_obs
from eve.util.coordtransform import vessel_cs_to_tracking3d


def anatomy_digest(tree):
    """Same geometry hashing convention as TD-MPC2 and Dreamer eval_seeds."""
    result = hashlib.sha256()
    for branch in sorted(tree.branches, key=lambda branch: branch.name):
        result.update(branch.name.encode() + b'\0')
        for value in (branch.coordinates, branch.radii):
            array = np.ascontiguousarray(value)
            result.update(str((array.shape, array.dtype.str)).encode())
            result.update(array.tobytes())
    return result.hexdigest()


def target_branch(intervention):
    target, tree = intervention.target, intervention.vessel_tree
    matches = []
    for name in target.branches if target.branches is not None else tree.keys():
        if name not in tree.keys():
            continue
        points = vessel_cs_to_tracking3d(tree[name].coordinates,
                                        target.fluoroscopy.image_rot_zx,
                                        target.fluoroscopy.image_center, None)
        if np.any(np.linalg.norm(points - target.coordinates3d, axis=1) <= 1e-4):
            matches.append(name)
    return matches[0] if len(matches) == 1 else ('ambiguous' if matches else 'unknown')


def evaluate(path, output, config, identities):
    output.mkdir()
    metadata = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                    config=config, status='running', completed_episodes=0,
                    recurrent_reset='head/body cleared per episode',
                    policy_action='existing deterministic SACPlayOnly.get_eval_action',
                    timing='wall seconds excluding reset; includes inference and environment steps',
                    success_rate_units='percent', empty_subset='null / blank CSV')
    write_json(output / 'evaluation.json', metadata)
    env = algo = None
    rows = []
    try:
        algo, step = load_policy(path, config['device'])
        metadata['checkpoint_step'] = step
        policy = algo.model.policy
        if policy.n_observations != 16 or policy.n_actions != 2 or type(policy.head).__name__ != 'LSTM':
            raise ValueError('Expected a state policy with 16 inputs, 2 actions and an LSTM head')
        intervention = ArchVarietyV2(arch_type=config['arch_type'],
                                   anatomy_seed=config['constructor_seed'],
                                   simulation_seed=config['simulation_seed'])
        # eval mode uses MaxSteps only: no VesselEnd truncation.
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
            algo.reset()
            policy.head.reset()
            policy.body.reset()
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
            row = dict(checkpoint_step=step, case_id=f'seed_{seed:06d}', environment_seed=seed,
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
            print(f'{path.name} | {index}/100 | seed={seed} | {branch} | '
                  f'success={int(row["success"])} steps={steps} path_ratio={ratio:.4f}', flush=True)
        summary = summarize(output, rows)
        metadata['status'] = 'complete'
        return dict(checkpoint=path.name, checkpoint_step=step, **summary['overall'])
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'evaluation.json', metadata)
        try:
            if env is not None:
                env.close()
        finally:
            if algo is not None:
                algo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/rnn_sac/eval_seeds.yaml')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--checkpoint', type=Path)
    choice.add_argument('--checkpoints-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--arch-type', choices=['I', 'II'], help='Overrides config arch_type')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    expected = {'arch_type', 'constructor_seed', 'simulation_seed', 'max_steps', 'device', 'threads', 'eval_seeds'}
    if not isinstance(config, dict) or set(config) != expected:
        parser.error(f'Expected config fields: {sorted(expected)}')
    if args.arch_type:
        config['arch_type'] = args.arch_type
    if config['arch_type'] not in ('I', 'II'):
        parser.error('arch_type must be I or II')
    for key in ('constructor_seed', 'simulation_seed', 'max_steps', 'threads'):
        if type(config[key]) is not int or config[key] < (1 if key in ('max_steps', 'threads') else 0):
            parser.error(f'Invalid {key}')
    seeds = config['eval_seeds']
    if (not isinstance(seeds, list) or len(seeds) != 100 or
            any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 100):
        parser.error('eval_seeds must contain exactly 100 unique nonnegative integers')
    torch.set_num_threads(config['threads'])
    if args.checkpoint:
        paths = [args.checkpoint.expanduser().resolve()]
    else:
        paths = sorted(args.checkpoints_dir.expanduser().resolve().glob('*.everl'),
                       key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)],
                       reverse=True)
    if not paths or any(not path.is_file() for path in paths):
        parser.error('No valid checkpoint files found')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(status='running', checkpoints=[dict(path=str(p), sha256=digest(p)) for p in paths],
                    config=config, completed_checkpoints=0)
    write_json(args.output / 'batch.json', manifest)
    summaries, identities = [], {}
    try:
        for path, entry in zip(paths, manifest['checkpoints']):
            if digest(path) != entry['sha256']:
                raise RuntimeError(f'Checkpoint changed since batch started: {path}')
            summaries.append(evaluate(path, args.output / path.stem, config, identities))
            write_csv(args.output / 'checkpoints_summary.csv', summaries)
            manifest['completed_checkpoints'] = len(summaries)
            write_json(args.output / 'batch.json', manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.output / 'case_identities.json', identities)
        write_json(args.output / 'batch.json', manifest)


if __name__ == '__main__':
    main()
