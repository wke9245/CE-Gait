"""Candidate-conditioned, permutation-invariant set attack, Eqs. (13)--(17)."""
import torch
from torch import nn


def relation_input(history, candidate):
    candidate = candidate.unsqueeze(1).expand(-1, history.shape[1], -1)
    return torch.cat([history, candidate, (history - candidate).abs(), history * candidate], -1)


class QuerySetAdversary(nn.Module):
    def __init__(self, dimension=256, hidden=256, dropout=0.2):
        super().__init__()
        self.relation = nn.Sequential(nn.Linear(4 * dimension, 512), nn.ReLU(),
                                      nn.Linear(512, hidden), nn.LayerNorm(hidden))
        self.classifier = nn.Sequential(nn.Linear(2 * hidden, 256), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, history, candidate, mask=None):
        if history.shape[1] == 0:
            return candidate.sum(-1) * 0
        if mask is None:
            mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        relation = self.relation(relation_input(history, candidate))
        count = mask.sum(1, keepdim=True)
        mean = (relation * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1)
        maximum = relation.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        logits = self.classifier(torch.cat([mean, maximum], -1)).squeeze(-1)
        return torch.where(count.squeeze(1) > 0, logits, logits * 0)
