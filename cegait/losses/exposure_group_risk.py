import math
import torch


def exposure_group_risk(risks, tau=0.2, mean_groups=False):
    risks = torch.stack(list(risks))
    if mean_groups:
        return risks.mean()
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * (torch.logsumexp(risks / tau, 0) - math.log(risks.numel()))
