"""Online image/goal adapter and closed-loop evaluation on explicit stEVE seeds."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import image_uint8, normalize_target
from .encoder import FrozenEncoder
from .train import Policy


class StEVEAdapter:
    """Matches generate_demo.CollectionEnv without depending on Dreamer/JAX."""
    def __init__(self, expected, max_steps=200):
        from eve_bench import ArchVarietyV2
        from eve_bench.steve_env import StEveEnv

        # mode=train preserves the collector's MaxSteps/VesselEnd/SimError truncations.
        self.env = StEveEnv(intervention=ArchVarietyV2(), obs='image',
                            mode='train', visualisation=False, n_max_steps=max_steps)
        try:
            self.low = np.asarray(self.env.action_space.low, np.float32)
            self.high = np.asarray(self.env.action_space.high, np.float32)
            if self.low.size != 2 or not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
                raise ValueError('Expected a single device with two finite action bounds')
            if np.any(self.high <= self.low):
                raise ValueError('Degenerate action bounds')
            for actual, key in [(self.low, 'action_low'), (self.high, 'action_high')]:
                if actual.shape != np.asarray(expected[key]).shape or not np.allclose(actual, expected[key]):
                    raise ValueError(f'Live {key} differs from demonstrations')
            if not np.isclose(self.env.intervention.fluoroscopy.image_frequency, expected['image_frequency']):
                raise ValueError('Live control frequency differs from demonstrations')
        except Exception:
            self.env.close()
            raise

    def observation(self, obs):
        intervention = self.env.intervention
        if intervention.simulation.simulation_error:
            raise ValueError('Simulation returned an invalid observation')
        pixels = intervention.fluoroscopy._coord_transform_tracking_to_image(
            np.asarray(intervention.target.coordinates2d).reshape(1, 2), is_vessel=False)
        return image_uint8(obs['image']), normalize_target(np.asarray(pixels).reshape(2))

    def reset(self, seed):
        obs, _ = self.env.reset(seed=int(seed))
        return self.observation(obs)

    def step(self, action):
        action = np.asarray(action, np.float32).reshape(self.low.shape)
        if not np.isfinite(action).all():
            raise ValueError('Policy returned nonfinite actions')
        command = (np.clip(action, -1, 1) + 1) / 2 * (self.high - self.low) + self.low
        obs, reward, terminated, truncated, info = self.env.step(command.copy())
        reasons = {type(x).__name__: bool(x.truncated) for x in self.env.truncation.truncations}
        done = bool(terminated or truncated)
        # A terminal simulation error need not supply a usable next image.
        next_obs = None if done else self.observation(obs)
        return next_obs, float(reward), done, {
            'success': bool(info.get('success', terminated)),
            'vessel_end': reasons.get('VesselEnd', False),
            'simulation_error': reasons.get('SimError', False),
            'timeout': reasons.get('MaxSteps', False)}

    def close(self):
        self.env.close()


class VisualPolicy:
    def __init__(self, encoder, checkpoint, device):
        self.encoder = encoder
        self.spec = checkpoint['spec']
        for key, value in encoder.spec.items():
            if self.spec[key] != value:
                raise ValueError(f'Encoder/cache mismatch: {key}')
        self.device = device
        self.policy = Policy().to(device).eval()
        self.policy.load_state_dict(checkpoint['policy'])
        self.frames = deque(maxlen=self.spec['window'])
        self.previous = np.zeros(2, np.float32)

    def reset(self):
        self.frames.clear()
        self.previous.fill(0)

    @torch.inference_mode()
    def act(self, image, target):
        if not self.frames:
            self.frames.extend([image.copy() for _ in range(self.frames.maxlen)])
        else:
            self.frames.append(image.copy())
        feature = self.encoder(np.stack(self.frames)[None])[0]
        inputs = np.concatenate([feature, target, self.previous]).astype(np.float32)
        action = self.policy(torch.from_numpy(inputs).to(self.device)[None])[0].cpu().numpy()
        self.previous = action.copy()
        return action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True, help='Same ViT-L checkpoint used for extraction')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='Explicit evaluation seeds; verify these are held out')
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.max_steps < 1 or len(args.seeds) != len(set(args.seeds)):
        p.error('max-steps must be positive and seeds must be unique')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.policy, map_location='cpu', weights_only=True)
    encoder = FrozenEncoder(args.repo, args.checkpoint, args.device)
    policy = VisualPolicy(encoder, checkpoint, args.device)
    adapter = StEVEAdapter(checkpoint['spec']['environment'], args.max_steps)
    rows = []
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    try:
        with open(output / 'episodes.jsonl', 'x') as stream:
            for seed in args.seeds:
                row = {'seed': seed, 'return': 0., 'steps': 0, 'success': False,
                       'vessel_end': False, 'simulation_error': False, 'timeout': False}
                inference = []
                try:
                    obs = adapter.reset(seed)
                    policy.reset()
                    for step in range(args.max_steps):
                        start = time.perf_counter()
                        action = policy.act(*obs)
                        inference.append(time.perf_counter() - start)
                        obs, reward, done, flags = adapter.step(action)
                        row.update(flags)
                        row['return'] += reward
                        row['steps'] = step + 1
                        if done:
                            break
                    else:
                        row['timeout'] = True
                except (ValueError, RuntimeError) as exc:
                    # Fail loudly: an adapter/model failure is not an environment outcome.
                    row['error'] = str(exc)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    raise
                row['mean_inference_seconds'] = float(np.mean(inference))
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(row, flush=True)
    finally:
        adapter.close()
    success_steps = [r['steps'] for r in rows if r['success']]
    summary = {key + '_rate': float(np.mean([r[key] for r in rows]))
               for key in ('success', 'vessel_end', 'simulation_error', 'timeout')}
    summary.update(episodes=len(rows), mean_return=float(np.mean([r['return'] for r in rows])),
                   mean_steps_to_success=float(np.mean(success_steps)) if success_steps else None,
                   mean_inference_seconds=float(np.average(
                       [r['mean_inference_seconds'] for r in rows], weights=[r['steps'] for r in rows])))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
