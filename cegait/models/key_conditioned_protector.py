"""CE-Gait Eqs. (5)--(11), exact dimensions and counts from §4.1.3."""
import torch
from torch import nn
from .quantization import quantize


class KeyConditionedProtector(nn.Module):
    def __init__(self, bits=4, early_key=True, local_remap=True, key_weighting=True):
        super().__init__()
        self.bits, self.early_key = bits, early_key
        self.local_remap, self.key_weighting = local_remap, key_weighting
        self.key_encoder = nn.Sequential(nn.Linear(256, 256, bias=False), nn.LayerNorm(256))
        self.local_norm = nn.LayerNorm(256)
        self.w1, self.u1 = nn.Linear(256, 512), nn.Linear(256, 512, bias=False)
        self.w2 = nn.Linear(512, 256)
        self.wa, self.ua = nn.Linear(256, 128), nn.Linear(256, 128, bias=False)
        self.score = nn.Linear(128, 1)
        self.phi = nn.Sequential(nn.Linear(256, 512), nn.LayerNorm(512),
                                 nn.LeakyReLU(0.1), nn.Linear(512, 512))
        self.head = nn.Sequential(nn.Linear(768, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.1),
                                  nn.Linear(1024, 512), nn.LayerNorm(512), nn.LeakyReLU(0.1),
                                  nn.Linear(512, 256))
        self.context_norm = nn.LayerNorm(256) if not early_key else nn.Identity()

    def forward(self, local, key):
        if local.ndim != 3 or local.shape[1:] != (16, 256):
            raise ValueError("IHGR local features must have shape [N,16,256]")
        if key.ndim == 1:
            key = key.unsqueeze(0)
        embedding = self.key_encoder(key.to(local)).expand(local.shape[0], -1)
        condition = embedding if self.early_key else self.context_norm(local.mean(1))
        remapped = local
        if self.local_remap:
            residual = self.w1(self.local_norm(local)) + self.u1(condition).unsqueeze(1)
            remapped = local + self.w2(torch.nn.functional.leaky_relu(residual, 0.1))
        c = condition if self.key_weighting else torch.zeros_like(condition)
        logits = self.score(torch.tanh(self.wa(remapped) + self.ua(c).unsqueeze(1)))
        attention = logits.softmax(1)
        aggregate = (attention * self.phi(remapped)).sum(1)
        continuous = torch.tanh(self.head(torch.cat([aggregate, embedding], -1)))
        exposed, codes = quantize(continuous, self.bits, self.training)
        return {"template": exposed, "codes": codes, "continuous": continuous,
                "attention": attention.squeeze(-1)}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
