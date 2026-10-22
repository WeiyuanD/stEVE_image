"""Shared offline/online preprocessing and frozen encoder loading."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenEncoder:
    def __init__(self, repo, checkpoint, device='cuda'):
        repo = Path(repo).resolve()
        checkpoint = Path(checkpoint).resolve()
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; explicitly use --device cpu for CPU inference')
        if not (repo / 'hubconf.py').is_file():
            raise ValueError(f'Not a V-JEPA2 checkout: {repo}')
        sys.path.insert(0, str(repo))
        from src.models.vision_transformer import vit_large
        from evals.hub.preprocessor import vjepa2_preprocessor

        self.device = torch.device(device)
        # Match src/hub/backbones.py without allocating its unused predictor.
        self.model = vit_large(img_size=(256, 256), patch_size=16,
                               num_frames=64, tubelet_size=2, use_sdpa=True,
                               use_SiLU=False, wide_SiLU=True,
                               uniform_power=False, use_rope=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['target_encoder']
        state = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state.items()}
        # The official checkpoint may contain fixed positional embeddings; this encoder uses RoPE.
        state.pop('pos_embed', None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.processor = vjepa2_preprocessor(crop_size=256)
        self.spec = {'model': 'vjepa2_vit_large', 'feature_dim': 1024,
                     'checkpoint_sha256': sha256(checkpoint),
                     'repo_commit': subprocess.check_output(
                         ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                     'preprocess': 'official-eval-256-gray-repeat-rgb-v1',
                     'pooling': 'mean-all-tokens', 'precision': 'float32'}

    @torch.inference_mode()
    def __call__(self, frames):
        frames = np.asarray(frames)
        if frames.ndim != 5 or frames.shape[2:] != (96, 96, 1) or frames.dtype != np.uint8:
            raise ValueError('Expected uint8 [B,T,96,96,1] frames')
        clips = [self.processor(list(np.repeat(clip, 3, axis=-1))) for clip in frames]
        tokens = self.model(torch.stack(clips).to(self.device))
        return tokens.mean(dim=1).float().cpu().numpy()
