"""Causal HDF5 windows; file handles are opened locally, never shared by workers."""
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


def metadata(handle):
    return json.loads(handle['metadata/json'][()])


def image_uint8(image):
    """Collector convention: grayscale intensities in [0,255], including float inputs."""
    image = np.asarray(image)
    if image.shape == (96, 96):
        image = image[..., None]
    if image.shape != (96, 96, 1):
        raise ValueError(f'Expected 96x96x1 grayscale image, got {image.shape}')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError('Images must contain finite intensities in [0,255]')
    return image.astype(np.uint8)


def normalize_target(pixel):
    pixel = np.asarray(pixel, dtype=np.float32)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError('Expected finite target pixel coordinates (x,y)')
    if np.any(pixel < 0) or np.any(pixel >= 96):
        raise ValueError('Target lies outside the recorded 96x96 image')
    return pixel / 48.0 - 1.0


def create_splits(root, output, seed=0, group_by='geometry_id'):
    """Split groups 80/10/10. Missing grouping metadata is an error, not a fallback."""
    root = Path(root).resolve()
    groups = {}
    for path in sorted((root / 'episodes').glob('*.h5')):
        with h5py.File(path, 'r') as handle:
            key = path.stem if group_by == 'episode' else metadata(handle).get(group_by)
        if key is None:
            raise ValueError(f'{path}: missing {group_by}; choose a grouping explicitly')
        groups.setdefault(str(key), []).append(str(path.relative_to(root)))
    if len(groups) < 3:
        raise ValueError('Need at least three groups for train/val/test splits')
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    n_val = max(1, round(len(keys) * .1))
    n_test = max(1, round(len(keys) * .1))
    partitions = {'val': keys[:n_val], 'test': keys[n_val:n_val+n_test],
                  'train': keys[n_val+n_test:]}
    result = {'dataset_root': str(root), 'seed': seed, 'group_by': group_by,
              'splits': {name: sorted(p for k in group for p in groups[k])
                         for name, group in partitions.items()}}
    with open(output, 'x') as stream:
        json.dump(result, stream, indent=2)
    return result


class HDF5Windows(Dataset):
    """Observation t and past frames predict normalized action t; no future frames."""

    def __init__(self, root, episodes, window=8):
        if window < 2 or window % 2:
            raise ValueError('V-JEPA windows must have a positive even length >=2')
        self.window = window
        self.paths = [Path(root) / p for p in episodes]
        self.index = []
        self.skipped = 0
        self.environment = None
        for episode, path in enumerate(self.paths):
            with h5py.File(path, 'r') as f:
                n = len(f['actions/normalized'])
                if f['observations/image'].shape != (n+1, 96, 96, 1):
                    raise ValueError(f'{path}: expected N+1 images for N transitions')
                if f['observations/image'].dtype != np.uint8:
                    raise ValueError(f'{path}: expected uint8 images')
                actions = f['actions/normalized'][:]
                if actions.shape != (n, 2):
                    raise ValueError(f'{path}: only single-device, two-action data supported')
                meta = metadata(f)
                env = {key: meta[key] for key in ('action_low', 'action_high', 'image_frequency')}
                if self.environment is None:
                    self.environment = env
                elif env != self.environment:
                    raise ValueError('Action bounds/image frequency differ between episodes')
                obs_valid = f['observations/valid'][:].astype(bool)
                targets = f['observations/target/pixel_coordinates'][:]
                target_valid = f['observations/target/pixel_valid'][:].astype(bool)
                valid = f['transitions/valid'][:].astype(bool)
                valid &= f['actions/applied_valid'][:].astype(bool)
                action_valid = np.isfinite(actions).all(1) & (np.abs(actions) <= 1).all(1)
                valid &= action_valid
                times = f['observations/time_sim_s'][:]
                dt = f['transitions/dt_s'][:]
                if not np.isfinite(dt).all() or np.any(dt <= 0) or not np.allclose(np.diff(times), dt):
                    raise ValueError(f'{path}: inconsistent transition timestamps')
                if not np.allclose(dt, 1.0 / float(env['image_frequency'])):
                    raise ValueError(f'{path}: unexpected action duration; repeated actions are unsupported')
                ends = f['transitions/episode_end'][:].astype(bool)
                if ends[:-1].any():
                    raise ValueError(f'{path}: contains an internal episode boundary')
                for t in range(n):
                    start = max(0, t-window+1)
                    good = valid[t] and obs_valid[start:t+2].all() and target_valid[t]
                    good &= np.isfinite(targets[t]).all() and ((targets[t] >= 0) & (targets[t] < 96)).all()
                    good &= valid[start:t].all()
                    if t:
                        good &= valid[t-1]
                    if good:
                        self.index.append((episode, t))
                    else:
                        self.skipped += 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, t = self.index[index]
        with h5py.File(self.paths[episode], 'r') as f:
            frames = f['observations/image'][max(0, t-self.window+1):t+1]
            if len(frames) < self.window:
                frames = np.concatenate([np.repeat(frames[:1], self.window-len(frames), axis=0), frames])
            target = normalize_target(f['observations/target/pixel_coordinates'][t])
            previous = f['actions/normalized'][t-1] if t else np.zeros(2, np.float32)
            return {'frames': frames, 'aux': np.concatenate([target, previous]).astype(np.float32),
                    'action': f['actions/normalized'][t].astype(np.float32),
                    'episode': episode, 'step': t}
