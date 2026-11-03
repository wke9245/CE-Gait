"""Priv-IHGR: local evidence sanitization and block-orthogonal keyed coding.

Local descriptor energy determines attenuation weights. The projection is
optimized with utility, quantization, and cross-key similarity objectives.
"""
import hashlib
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..models.quantization import quantize


class PrivIHGR(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.projection = nn.Linear(256, 256)
        self.cache = OrderedDict()

    def orthogonal(self, vector):
        raw = (vector.detach().cpu().numpy() > 0).astype(np.uint8).tobytes()
        digest = hashlib.sha256(raw).digest()
        token = (digest, str(vector.device))
        if token not in self.cache:
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            blocks = [torch.tensor(np.linalg.qr(rng.normal(size=(32, 32)))[0],
                                   device=vector.device, dtype=torch.float32) for _ in range(8)]
            self.cache[token] = torch.block_diag(*blocks)
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        return self.cache[token]

    def forward(self, local, key):
        if key.ndim == 1:
            key = key[None]
        key = key.expand(len(local), -1)
        saliency = local.abs().mean(-1)
        importance = saliency / saliency.amax(1, keepdim=True).clamp_min(1e-8)
        refined = local * (1 - 0.3 * importance.unsqueeze(-1))
        if self.training:
            refined = refined + torch.randn_like(refined) * refined.std(dim=(1, 2), keepdim=True) * 0.05
        pooled = self.projection(refined.mean(1))
        continuous = torch.empty_like(pooled)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True)
        for i, vector in enumerate(unique):
            selected = inverse == i
            continuous[selected] = torch.tanh(pooled[selected] @ self.orthogonal(vector))
        template, codes = quantize(continuous, self.bits, self.training)
        return {"template": template, "continuous": continuous, "codes": codes, "attention": importance}

    @staticmethod
    def cross_key_loss(first, second):
        return F.relu(F.cosine_similarity(first, second) - 0.1).mean()
