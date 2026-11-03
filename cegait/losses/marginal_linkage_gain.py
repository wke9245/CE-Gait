from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def frozen_adversary(adversary):
    state, flags = adversary.training, [p.requires_grad for p in adversary.parameters()]
    adversary.eval()  # identical paired predictions, no independent dropout noise
    for p in adversary.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(adversary.parameters(), flags):
            p.requires_grad_(flag)
        adversary.train(state)


def sample_context_masks(batch, length, contexts, generator=None, device="cpu"):
    before, after = [], []
    for record in range(length):
        other = [i for i in range(length) if i != record]
        for _ in range(contexts):
            mask = torch.zeros(batch, length, dtype=torch.bool)
            for row in range(batch):
                size = int(torch.randint(length, (1,), generator=generator))
                if size:
                    chosen = torch.randperm(length - 1, generator=generator)[:size]
                    mask[row, torch.tensor(other)[chosen]] = True
            extended = mask.clone()
            extended[:, record] = True
            before.append(mask)
            after.append(extended)
    return torch.stack(before).to(device), torch.stack(after).to(device)


def marginal_linkage_gain(adversary, history, candidate, labels, contexts=4,
                         positive_part=True, generator=None, return_details=False):
    n, m, d = history.shape
    if m < 1 or contexts < 1:
        raise ValueError("Nonempty history and positive context count required")
    before, after = sample_context_masks(n, m, contexts, generator, history.device)
    pairs = before.shape[0]
    h = history.unsqueeze(0).expand(pairs, -1, -1, -1).reshape(-1, m, d)
    c = candidate.unsqueeze(0).expand(pairs, -1, -1).reshape(-1, d)
    y = labels.unsqueeze(0).expand(pairs, -1).reshape(-1)
    pre = F.binary_cross_entropy_with_logits(adversary(h, c, before.reshape(-1, m)), y, reduction="none")
    post = F.binary_cross_entropy_with_logits(adversary(h, c, after.reshape(-1, m)), y, reduction="none")
    gains = (pre - post).reshape(pairs, n)
    loss = gains.clamp_min(0).mean() if positive_part else gains.mean()
    return (loss, gains) if return_details else loss


def confusion_loss(logits):
    return 0.5 * (F.softplus(logits) + F.softplus(-logits)) - torch.log(logits.new_tensor(2.0))
