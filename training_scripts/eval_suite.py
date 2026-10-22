"""Evaluate state RNN-SAC checkpoints on an archived stEVE suite (no learning)."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from time import perf_counter

# Use the checkout containing the trained network definitions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eve_rl'))

import numpy as np
import torch
import yaml
from eve.truncation import MaxSteps
from eve_bench.cases import CaseSuite
from eve_bench.steve_env import make_state_observation
from eve_rl.util import ConfigHandler, flatten_obs


DEFAULT_SUITE = '/home/mtgroup/WeiyuanD/11_dataset/evaluation_sets/type_I/200_cases_20_anatomies_10_targets'
DEFAULT_CHECKPOINTS = '/home/mtgroup/WeiyuanD/stEVE_training/results/rnn_sac_state_150_450/best_ckpt'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.csv.partial')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(rows):
    successes = [row for row in rows if row['success']]
    failures = [row for row in rows if not row['success']]
    def average(group, key):
        return float(np.mean([row[key] for row in group])) if group else None
    return dict(episodes=len(rows), success_rate=100.0 * len(successes) / len(rows),
                mean_navigation_time=average(successes, 'wall_time_s'),
                mean_path_ratio_unsuccessful=average(failures, 'final_path_ratio'),
                navigation_time_samples=len(successes), path_ratio_samples=len(failures))


def summarize(output, rows):
    summary = {'overall': aggregate(rows)}
    for name, key in (('by_anatomy', 'anatomy_id'), ('by_branch', 'target_branch')):
        summary[name] = {identity: aggregate([r for r in rows if r[key] == identity])
                         for identity in sorted({r[key] for r in rows})}
    write_json(output / 'summary.json', summary)
    write_csv(output / 'summary_overall.csv', [summary['overall']])
    write_csv(output / 'summary_by_anatomy.csv',
              [dict(anatomy_id=k, **v) for k, v in summary['by_anatomy'].items()])
    write_csv(output / 'summary_by_target.csv',
              [dict(target_branch=k, **v) for k, v in summary['by_branch'].items()])
    return summary


def load_policy(path, device):
    # Trusted local checkpoints include Python config metadata, not only tensors.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    training_algo = ConfigHandler().config_dict_to_object(checkpoint['algo'])
    training_algo.load_state_dicts_network(checkpoint['network_state_dicts'])
    algo = training_algo.to_play_only()
    del training_algo
    algo.to(torch.device(device))
    algo.stochastic_eval = False
    algo.model.policy.eval()
    return algo, int(checkpoint['steps']['exploration'])


def evaluate_checkpoint(path, output, suite, args, expected_digest):
    output.mkdir()
    if digest(path) != expected_digest:
        raise RuntimeError(f'Checkpoint changed since batch started: {path}')
    identity = dict(checkpoint=str(path), checkpoint_sha256=expected_digest,
                    suite_dir=str(suite.directory), suite_id=suite.suite_id,
                    max_steps=args.max_steps, vessel_end_truncation=False,
                    observation='state', stochastic_eval=False,
                    recurrent_reset='clear policy head and body at each case',
                    policy_action='existing SACPlayOnly.get_eval_action',
                    timing='wall seconds after reset/state setup, including inference and environment steps',
                    success_rate_units='percent', empty_subset='null / blank CSV',
                    status='running', completed_cases=0)
    write_json(output / 'evaluation.json', identity)
    algo = None
    rows = []
    try:
        algo, checkpoint_step = load_policy(path, args.device)
        identity['checkpoint_step'] = checkpoint_step
        for entry in suite.manifest['cases']:
            case_id = entry['case_id']
            raw = suite.load(case_id)
            try:
                base = raw.unwrapped
                if base.intervention.normalize_action:
                    raise ValueError('Archived suite must expose physical actions')
                low = np.asarray(raw.action_space.low, dtype=np.float32)
                high = np.asarray(raw.action_space.high, dtype=np.float32)
                if low.size != 2 or not np.isfinite([low, high]).all() or not (high > low).all():
                    raise ValueError('Expected finite two-component physical action bounds')
                # Do not replace the archived observation: CaseEnv.reset verifies it.
                # Replace ALL old truncations, including its old MaxSteps/VesselEnd.
                base.truncation = MaxSteps(args.max_steps)
                state = make_state_observation(base.intervention)
                raw.reset()
                state.reset(base.episode_number - 1)
                policy = algo.model.policy
                algo.reset()
                # Upstream policy.reset is a no-op; explicitly isolate case history.
                policy.head.reset()
                policy.body.reset()
                total_return = 0.0
                terminal = truncated = False
                start = perf_counter()
                for step in range(1, args.max_steps + 1):
                    flat, _ = flatten_obs(state())
                    if flat.size != policy.n_observations or not np.isfinite(flat).all():
                        raise ValueError('State dimensions/values do not match checkpoint')
                    with torch.inference_mode():
                        action = np.asarray(algo.get_eval_action(flat), dtype=np.float32)
                    if action.size != low.size or not np.isfinite(action).all():
                        raise ValueError('Invalid policy action')
                    action = action.reshape(low.shape)
                    # Exactly one normalized-to-physical conversion, as in Single.
                    physical = (action + 1) / 2 * (high - low) + low
                    _, reward, terminal, truncated, info = raw.step(physical)
                    state.step()
                    total_return += float(reward)
                    if terminal or truncated:
                        break
                elapsed = perf_counter() - start
                ratio = float(info['path_ratio'])
                if not np.isfinite(ratio) or not np.isfinite(total_return):
                    raise ValueError('Nonfinite episode metrics')
                row = dict(checkpoint_step=checkpoint_step, case_id=case_id,
                           anatomy_id=entry['anatomy_id'], target_branch=entry['target_branch'],
                           success=bool(info['success']), steps=step,
                           **{'return': total_return}, final_path_ratio=ratio,
                           wall_time_s=elapsed, terminated=bool(terminal), truncated=bool(truncated))
                rows.append(row)
                # Persist each completed episode even if a later case fails.
                with (output / 'episodes.csv').open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if len(rows) == 1:
                        writer.writeheader()
                    writer.writerow(row)
                with (output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                identity['completed_cases'] = len(rows)
                write_json(output / 'evaluation.json', identity)
                print(f'{path.name} | {len(rows)}/{len(suite.case_ids)} | {case_id} | '
                      f'success={int(row["success"])} steps={step} path_ratio={ratio:.4f}', flush=True)
            finally:
                raw.close()
        summary = summarize(output, rows)
        identity['status'] = 'complete'
        return checkpoint_step, summary['overall']
    except BaseException as exc:
        identity.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'evaluation.json', identity)
        if algo is not None:
            algo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=Path(DEFAULT_SUITE))
    parser.add_argument('--checkpoints-dir', type=Path, default=Path(DEFAULT_CHECKPOINTS))
    parser.add_argument('--training-config', type=Path, help='Defaults to checkpoint folder parent/config.yaml')
    parser.add_argument('--pattern', default='*.everl')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--max-steps', type=int, default=450)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.max_steps < 1 or args.threads < 1:
        parser.error('max-steps and threads must be positive')
    torch.set_num_threads(args.threads)
    root = args.checkpoints_dir.expanduser().resolve()
    paths = sorted((p for p in root.glob(args.pattern) if p.is_file()),
                   key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)],
                   reverse=True)  # checkpoint<step>.everl: newest training step first.
    if not paths:
        parser.error(f'No checkpoints matching {args.pattern} in {root}')
    config_path = args.training_config or root.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    if config['env']['obs'] != 'state':
        parser.error('This evaluator supports the current state-based checkpoints only')
    suite = CaseSuite(args.suite)
    entries = suite.manifest['cases']
    if (len(entries), len(suite.anatomy_ids), len({r['target_branch'] for r in entries})) != (200, 20, 5):
        parser.error('Expected 200 cases, 20 anatomies and 5 target branches')
    for case_id in suite.case_ids:
        suite.case_metadata(case_id)
    checkpoints = [(p, digest(p)) for p in paths]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'training_config.yaml').write_text(config_path.read_text())
    manifest = dict(suite_id=suite.suite_id, suite_dir=str(suite.directory),
                    max_steps=args.max_steps, device=args.device, threads=args.threads,
                    checkpoints=[dict(path=str(p), sha256=h) for p, h in checkpoints], status='running')
    write_json(args.output / 'batch.json', manifest)
    results = []
    try:
        for path, checksum in checkpoints:
            result = dict(checkpoint=path.name, checkpoint_step=None, status='failed', error='',
                          episodes=None, success_rate=None, mean_navigation_time=None,
                          mean_path_ratio_unsuccessful=None, navigation_time_samples=None,
                          path_ratio_samples=None)
            try:
                step, summary = evaluate_checkpoint(path, args.output / path.stem, suite, args, checksum)
                result.update(checkpoint_step=step, status='complete', **summary)
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
                logging.exception('Checkpoint evaluation failed: %s', path)
            results.append(result)
            write_csv(args.output / 'checkpoints_summary.csv', results)
        manifest['status'] = 'complete' if all(r['status'] == 'complete' for r in results) else 'completed_with_errors'
    except BaseException:
        manifest['status'] = 'interrupted'
        raise
    finally:
        manifest['completed_checkpoints'] = sum(r['status'] == 'complete' for r in results)
        write_json(args.output / 'batch.json', manifest)
    if manifest['status'] != 'complete':
        raise SystemExit('Some checkpoints failed; see batch.json and per-checkpoint evaluation.json')


if __name__ == '__main__':
    main()
