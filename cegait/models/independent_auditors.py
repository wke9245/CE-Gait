"""Independently initialized DeepSets and SAB/PMA auditors."""
import torch
from torch import nn
from .query_set_adversary import relation_input


class DeepSetsAuditor(nn.Module):
    def __init__(self, dimension=256, hidden=256, disclose_keys=False):
        super().__init__()
        self.disclose_keys = disclose_keys
        self.encoder = nn.Sequential(nn.Linear(4 * dimension + (257 if disclose_keys else 0), hidden),
                                      nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, history, candidate, mask=None, key_slots=None):
        if history.shape[1] == 0:
            return candidate.sum(-1) * 0
        x = relation_input(history, candidate)
        if self.disclose_keys:
            if key_slots is None:
                key_slots = x.new_zeros(*x.shape[:2], 257)
            x = torch.cat([x, key_slots], -1)
        x = self.encoder(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        count = mask.sum(1, keepdim=True)
        result = self.decoder((x * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1)).squeeze(-1)
        return torch.where(count.squeeze(1) > 0, result, result * 0)


class SetTransformerAuditor(nn.Module):
    def __init__(self, dimension=256, hidden=128, heads=4, disclose_keys=False):
        super().__init__()
        self.disclose_keys = disclose_keys
        self.input = nn.Linear(4 * dimension + (257 if disclose_keys else 0), hidden)
        block = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.pool = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, history, candidate, mask=None, key_slots=None):
        if history.shape[1] == 0:
            return candidate.sum(-1) * 0
        x = relation_input(history, candidate)
        if self.disclose_keys:
            if key_slots is None:
                key_slots = x.new_zeros(*x.shape[:2], 257)
            x = torch.cat([x, key_slots], -1)
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        nonempty = valid.any(1)
        safe = valid.clone()
        safe[~nonempty, 0] = True
        x = self.encoder(self.input(x), src_key_padding_mask=~safe)
        seed = self.seed.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(seed, x, x, key_padding_mask=~safe, need_weights=False)
        result = self.head(self.norm(pooled[:, 0] + seed[:, 0])).squeeze(-1)
        return torch.where(nonempty, result, result * 0)


def make_auditor(kind, disclose_keys=False, dimension=256):
    if kind == "query":
        if disclose_keys:
            raise ValueError("Key slots require deepsets or transformer")
        from .query_set_adversary import QuerySetAdversary
        return QuerySetAdversary(dimension=dimension)
    if kind == "deepsets":
        return DeepSetsAuditor(dimension=dimension, disclose_keys=disclose_keys)
    if kind == "transformer":
        return SetTransformerAuditor(dimension=dimension, disclose_keys=disclose_keys)
    raise ValueError(f"Unknown auditor: {kind}")
