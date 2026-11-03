"""Storage-matched biometric hashing references, independently implemented."""
import hashlib
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..models.quantization import quantize


class RandomizedHash(nn.Module):
    def __init__(self, kind="mlp_hash", bits=4):
        super().__init__()
        self.kind, self.bits = kind, bits
        self.feature_level = True
        self.matching = "hamming" if kind in ("biohash", "rand_hash") else "collision" if kind in ("iom_grp", "iom_urp") else "cosine"
        self.cache = OrderedDict()

    def random_parameters(self, vector, dimension, device):
        raw = (vector.detach().cpu().numpy() > 0).astype(np.uint8).tobytes()
        fingerprint = hashlib.sha256(raw).digest()
        token = (fingerprint, dimension, str(device))
        if token in self.cache:
            self.cache.move_to_end(token)
            return self.cache[token]
        rng = np.random.default_rng(int.from_bytes(fingerprint[:8], "big"))
        def matrix(shape):
            return torch.tensor(rng.normal(size=shape) / np.sqrt(shape[0]), device=device, dtype=torch.float32)
        if self.kind == "biohash":
            params = matrix((dimension, 1024))
        elif self.kind == "iom_grp":
            params = matrix((dimension, 256 * 16)).reshape(dimension, 256, 16)
        elif self.kind == "iom_urp":
            params = torch.tensor(np.stack([[rng.permutation(dimension)[:16] for _ in range(2)]
                                            for _ in range(256)]), device=device)
        elif self.kind == "rand_hash":
            params = (torch.tensor(np.stack([rng.permutation(dimension) for _ in range(4)]), device=device),
                      torch.tensor(rng.uniform(0.5, 1.5, (4, dimension)), device=device, dtype=torch.float32),
                      torch.tensor(rng.choice([-1., 1.], (4, dimension)), device=device, dtype=torch.float32))
        elif self.kind == "simple_repair":
            q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            params = torch.tensor(q.astype(np.float32), device=device)
        elif self.kind == "mlp_hash":
            params = [matrix((dimension, 512)), matrix((512, 512)), matrix((512, 256))]
        else:
            raise ValueError(self.kind)
        self.cache[token] = params
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return params

    def forward(self, local, key):
        pooled = F.normalize(local.mean(1) if local.ndim == 3 else local, dim=-1)
        if key.ndim == 1:
            key = key[None]
        key = key.expand(len(local), -1)
        values = []
        # Group by key so a domain matrix is generated once, shared across identities.
        unique, inverse = torch.unique(key, dim=0, return_inverse=True)
        template = pooled.new_empty(len(local), 1024 if self.kind in ("biohash", "rand_hash") else 256)
        continuous = pooled.new_zeros(len(local), 256)
        codes = torch.empty_like(template, dtype=torch.uint8)
        for index, vector in enumerate(unique):
            selected = inverse == index
            x = pooled[selected]
            params = self.random_parameters(vector, x.shape[-1], x.device)
            if self.kind == "biohash":
                z = (x @ params >= 0).to(torch.uint8)
                v = z.float() * 2 - 1
            elif self.kind == "iom_grp":
                z = torch.einsum("nd,dhk->nhk", x, params).argmax(-1).to(torch.uint8)
                v = z.float() * (2 / 15) - 1
            elif self.kind == "iom_urp":
                products = x[:, params[:, 0]] * x[:, params[:, 1]]
                z = products.argmax(-1).to(torch.uint8)
                v = z.float() * (2 / 15) - 1
            elif self.kind == "rand_hash":
                permutation, scale, sign = params
                z = (x[:, permutation] * scale * sign >= 0).flatten(1).to(torch.uint8)
                v = z.float() * 2 - 1
            else:
                if self.kind == "simple_repair":
                    raw = torch.tanh(x @ params * np.sqrt(x.shape[-1]))
                else:
                    raw = x
                    for layer, weight in enumerate(params):
                        raw = raw @ weight
                        raw = F.leaky_relu(raw, 0.1) if layer < 2 else torch.tanh(raw * 4)
                v, z = quantize(raw, self.bits, False)
                continuous[selected] = raw
            template[selected], codes[selected] = v, z
        return {"template": template, "codes": codes, "continuous": continuous, "attention": local.new_zeros(len(local), 16)}


class MLPHashCM(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.feature_level = True
        self.key_encoder = nn.Sequential(nn.Linear(256, 256, bias=False), nn.LayerNorm(256))
        self.network = nn.Sequential(nn.Linear(512, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.1),
                                      nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.1),
                                      nn.Linear(1024, 560), nn.LayerNorm(560), nn.LeakyReLU(0.1),
                                      nn.Linear(560, 256))

    def forward(self, local, key):
        if key.ndim == 1:
            key = key[None]
        e = self.key_encoder(key).expand(local.shape[0], -1)
        pooled = local.mean(1) if local.ndim == 3 else local
        v = torch.tanh(self.network(torch.cat([pooled, e], -1)))
        t, codes = quantize(v, self.bits, self.training)
        return {"template": t, "codes": codes, "continuous": v, "attention": local.new_zeros(len(local), 16)}


class Unprotected(nn.Module):
    feature_level = True

    def forward(self, local, key):
        pooled = local.mean(1) if local.ndim == 3 else local
        return {"template": pooled, "continuous": pooled, "codes": pooled,
                "attention": pooled.new_zeros(len(pooled), 16)}


def make_protector(kind="cegait", **kwargs):
    if kind == "unprotected":
        return Unprotected()
    if kind == "priv_ihgr":
        from .priv_ihgr import PrivIHGR
        return PrivIHGR(bits=kwargs.get("bits", 4))
    if kind == "cegait":
        from ..models.key_conditioned_protector import KeyConditionedProtector
        return KeyConditionedProtector(**kwargs)
    if kind == "late_key_cm":
        from .late_key_cm import LateKeyCM
        return LateKeyCM(**kwargs)
    if kind == "mlp_hash_cm":
        return MLPHashCM(bits=kwargs.get("bits", 4))
    return RandomizedHash(kind, kwargs.get("bits", 4))
