import torch
import torch.nn.functional as F


def cosine_distance(x, y):
    return (1 - F.cosine_similarity(x, y, dim=-1)) / 2


def batch_hard_triplet(templates, identities, margin=0.2):
    distance = (1 - F.normalize(templates, dim=-1) @ F.normalize(templates, dim=-1).T) / 2
    same = identities[:, None].eq(identities[None, :])
    same.fill_diagonal_(False)
    different = ~identities[:, None].eq(identities[None, :])
    valid = same.any(1) & different.any(1)
    if not valid.any():
        raise ValueError("Utility batches require >=2 identities and >=2 clips each")
    pos = distance.masked_fill(~same, -torch.inf).amax(1)
    neg = distance.masked_fill(~different, torch.inf).amin(1)
    return F.relu(pos[valid] - neg[valid] + margin).mean()


def quantization_loss(continuous, exposed):
    return (continuous - exposed.detach()).square().sum(-1).mean()
