"""Train an MLP on cached frozen features; the test partition is not opened."""
import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1028, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 2), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.path = str(path)
        with h5py.File(path, 'r') as f:
            if not f.attrs.get('complete', False):
                raise ValueError(f'Incomplete feature cache: {path}')
            self.spec = json.loads(f.attrs['spec'])
            self.length = len(f['actions'])
        if not self.length:
            raise ValueError('Empty feature cache')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            return np.concatenate([f['features'][index], f['aux'][index]]), f['actions'][index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('epochs, batch-size and lr must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    datasets = {name: FeatureDataset(Path(args.cache) / f'{name}.h5') for name in ('train', 'val')}
    if datasets['train'].spec != datasets['val'].spec:
        raise ValueError('Training and validation feature specifications differ')
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=name == 'train',
                               num_workers=args.workers) for name, data in datasets.items()}
    policy = Policy().to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best = float('inf')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    with open(output / 'metrics.jsonl', 'x') as log:
        for epoch in range(1, args.epochs+1):
            metrics = {'epoch': epoch}
            for name, loader in loaders.items():
                training = name == 'train'
                policy.train(training)
                squared_error = torch.zeros(2, device=args.device)
                count = 0
                with torch.set_grad_enabled(training):
                    for inputs, target in loader:
                        inputs, target = inputs.to(args.device), target.to(args.device)
                        error = (policy(inputs) - target).square()
                        loss = error.mean()
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite policy loss')
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        squared_error += error.detach().sum(0)
                        count += len(inputs)
                mse = (squared_error / count).cpu().tolist()
                metrics[name + '_mse_per_action'] = mse
                metrics[name + '_mse'] = sum(mse) / 2
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            print(metrics, flush=True)
            if metrics['val_mse'] < best:
                best = metrics['val_mse']
                torch.save({'policy': policy.state_dict(), 'spec': datasets['train'].spec,
                            'epoch': epoch, 'val_mse': best, 'train_config': vars(args)},
                           output / 'best.pt')


if __name__ == '__main__':
    main()
