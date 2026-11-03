"""IHGR-Net: hierarchical spatial reliability and temporal attention.

Frame encoding uses six shared 3x3 convolutions with 64/128/256 channels.
SGRA operates at 1/2/4 spatial granularities; SKA supplies temporal pooling.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 1
        for stage, channels in enumerate((64, 128, 256)):
            for _ in range(2):
                layers += [nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.LeakyReLU(0.1)]
                previous = channels
            if stage < 2:
                layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class SGRA(nn.Module):
    """E=GAP(abs(F)); omega=sigmoid(alpha_s*E+beta_s), per part."""
    def __init__(self, levels=(1, 2, 4)):
        super().__init__()
        self.levels = tuple(levels)
        self.alpha_s = nn.Parameter(torch.tensor(1.0))
        self.beta_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, feature):
        # [N,T,C,H,W]; exact floor boundaries of paper Eq. (2)
        height = feature.shape[-2]
        maps, reliabilities, distributions = [], {}, {}
        for parts in self.levels:
            strips, scores = [], []
            for part in range(parts):
                start, end = part * height // parts, (part + 1) * height // parts
                strip = feature[..., start:end, :]
                energy = strip.abs().mean(dim=(-3, -2, -1))
                reliability = torch.sigmoid(self.alpha_s * energy + self.beta_s)
                strips.append(strip * reliability[..., None, None, None])
                scores.append(reliability)
            omega = torch.stack(scores, -1)
            reliabilities[str(parts)] = omega
            distributions[str(parts)] = omega / omega.sum(-1, keepdim=True).clamp_min(1e-8)
            maps.append(torch.cat(strips, -2))
        return torch.stack(maps).mean(0), reliabilities, distributions


class SKA(nn.Module):
    """Paper Eqs. (9)--(13): frame MLP, temporal softmax, weighted sum."""
    def __init__(self, channels=256, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, frame_descriptors, valid=None):
        score = self.scorer(frame_descriptors).squeeze(-1)
        if valid is not None:
            if not valid.any(1).all():
                raise ValueError("Each sequence must contain at least one valid frame")
            score = score.masked_fill(~valid, -torch.inf)
        attention = score.softmax(1)
        sequence = (frame_descriptors * attention.unsqueeze(-1)).sum(1)
        return sequence, attention, score


def entropy(probability):
    return -(probability * probability.clamp_min(1e-8).log()).sum(-1)


class IHGRNet(nn.Module):
    def __init__(self, num_classes=74, spatial_parts=4, temporal_segments=4):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_parts, self.temporal_segments = spatial_parts, temporal_segments
        self.encoder = FrameEncoder()
        self.sgra = SGRA()
        self.ska = SKA()
        self.classifier = nn.Linear(256, num_classes, bias=False)

    def forward(self, silhouettes, valid=None):
        if silhouettes.ndim == 4:
            silhouettes = silhouettes.unsqueeze(2)
        if silhouettes.ndim != 5 or silhouettes.shape[2] != 1:
            raise ValueError("silhouettes must have shape [N,T,1,64,44] or [N,T,64,44]")
        n, t, c, h, w = silhouettes.shape
        if t < self.temporal_segments:
            raise ValueError("At least four sampled frames are required")
        encoded = self.encoder(silhouettes.reshape(n * t, c, h, w))
        encoded = encoded.reshape(n, t, *encoded.shape[1:])
        refined, reliability, spatial_weights = self.sgra(encoded)
        frame = refined.mean((-2, -1))
        sequence, attention, scores = self.ska(frame, valid)
        # Expose local evidence before global temporal aggregation to CE-Gait.
        # Each segment uses the same SKA scorer, renormalized inside the segment.
        local = []
        height = refined.shape[-2]
        for p in range(self.spatial_parts):
            part = refined[..., p * height // self.spatial_parts:(p + 1) * height // self.spatial_parts, :]
            descriptor = part.mean((-2, -1))
            for segment in range(self.temporal_segments):
                begin, end = segment * t // self.temporal_segments, (segment + 1) * t // self.temporal_segments
                segment_scores = scores[:, begin:end]
                all_invalid = ~torch.isfinite(segment_scores).any(1)
                segment_scores = torch.where(all_invalid[:, None], torch.zeros_like(segment_scores), segment_scores)
                weights = segment_scores.softmax(1)
                pooled = (descriptor[:, begin:end] * weights.unsqueeze(-1)).sum(1)
                local.append(torch.where(all_invalid[:, None], torch.zeros_like(pooled), pooled))
        spatial_entropy = torch.stack([entropy(weights.mean(1)) for weights in spatial_weights.values()]).mean()
        sparsity = spatial_entropy + entropy(attention).mean()
        return {"embedding": sequence, "local": torch.stack(local, 1),
                "logits": self.classifier(sequence), "reliability": reliability,
                "spatial_weights": spatial_weights, "temporal_attention": attention,
                "key_frame": attention.argmax(1), "sparsity_loss": sparsity}

    @staticmethod
    def training_loss(output, labels, sparsity_weight=0.5):
        from ..losses.utility_loss import batch_hard_triplet
        triplet = batch_hard_triplet(output["embedding"], labels, 0.2)
        classification = F.cross_entropy(output["logits"], labels)
        return triplet + classification + sparsity_weight * output["sparsity_loss"]
