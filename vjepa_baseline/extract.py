"""python -m vjepa_baseline.extract --help"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

from .data import HDF5Windows, create_splits
from .encoder import FrozenEncoder, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split-file', help='Reuse an existing split manifest')
    p.add_argument('--group-by', choices=['geometry_id', 'seed', 'episode'], default='geometry_id')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text())
        if Path(split['dataset_root']).resolve() != Path(args.dataset).resolve():
            raise ValueError('Split manifest belongs to a different dataset')
        (output / 'splits.json').write_text(json.dumps(split, indent=2))
    else:
        split = create_splits(args.dataset, output / 'splits.json', args.seed, args.group_by)
    entries = [p for part in split['splits'].values() for p in part]
    if len(entries) != len(set(entries)):
        raise ValueError('Duplicate episodes across splits')
    encoder = FrozenEncoder(args.repo, args.checkpoint, args.device)
    environment = None
    for name in ('train', 'val', 'test'):
        dataset = HDF5Windows(args.dataset, split['splits'][name], args.window)
        if not len(dataset):
            raise ValueError(f'{name}: no valid windows')
        if environment is not None and environment != dataset.environment:
            raise ValueError('Environment settings differ across splits')
        environment = dataset.environment
        spec = dict(encoder.spec, window=args.window, environment=environment,
                    target_normalization='pixel_xy / 48 - 1',
                    split_sha256=sha256(output / 'splits.json'))
        print(f'{name}: {len(dataset)} windows; {dataset.skipped} filtered', flush=True)
        temporary = output / f'{name}.partial.h5'
        with h5py.File(temporary, 'x') as f:
            f.attrs['spec'] = json.dumps(spec)
            f.create_dataset('episode_paths', data=split['splits'][name],
                             dtype=h5py.string_dtype('utf-8'))
            arrays = {key: f.create_dataset(key, (len(dataset), width), dtype='f4', chunks=True)
                      for key, width in [('features', 1024), ('aux', 4), ('actions', 2)]}
            arrays.update({key: f.create_dataset(key, (len(dataset),), dtype='i8')
                           for key in ('episode', 'step')})
            offset = 0
            for batch in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
                features = encoder(batch['frames'].numpy())
                if not np.isfinite(features).all():
                    raise ValueError('Encoder returned nonfinite features')
                end = offset + len(features)
                arrays['features'][offset:end] = features
                for dest, src in [('aux', 'aux'), ('actions', 'action'), ('episode', 'episode'), ('step', 'step')]:
                    arrays[dest][offset:end] = batch[src].numpy()
                offset = end
                if offset % 100 == 0 or offset == len(dataset):
                    print(f'{name}: {offset}/{len(dataset)}', flush=True)
            f.attrs['complete'] = True
        temporary.rename(output / f'{name}.h5')


if __name__ == '__main__':
    main()
