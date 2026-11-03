"""Full-gallery subgroup recognition and within-group identity resampling."""
import numpy as np
import torch
from scipy.stats import fisher_exact
from ..protocols.history_episode_builder import HistoryEpisodeBuilder
from ..protocols.domain_key_schedule import DomainKeySchedule
from .recognition import rank_metrics


@torch.no_grad()
def subgroup_recognition(model, store, schedules=10, seed=42, device="cpu", resamples=10000):
    if store.dataset != "gait3d":
        return {}
    rows = [r for r in store.records if r["role"] == "test"]
    identities = sorted({str(r["identity"]) for r in rows})
    eligible = set(HistoryEpisodeBuilder(store, seed).eligible("test", "seq", 4))
    is_eligible = np.array([identity in eligible for identity in identities])
    if is_eligible.all() or not is_eligible.any():
        return {"eligible_identities": len(eligible), "other_identities": len(identities) - len(eligible)}
    correctness = np.zeros((schedules, len(identities)))
    matching = getattr(model, "matching", "cosine")
    indices = [r["index"] for r in rows]
    gallery = np.array([r["gallery"] for r in rows])
    labels = np.array([str(r["identity"]) for r in rows])
    model.eval()
    for schedule in range(schedules):
        key = DomainKeySchedule(seed, "recognition/test").vectors(schedule, ["active"], device)
        values = []
        for start in range(0, len(rows), 128):
            result = model(store.tensor(indices[start:start + 128], device, getattr(model, "feature_level", False)), key)
            values.append(result["codes" if matching != "cosine" else "template"].cpu().numpy())
        values = np.concatenate(values)
        for index, identity in enumerate(identities):
            probe = (~gallery) & (labels == identity)
            correctness[schedule, index] = rank_metrics(values[probe], values[gallery], labels[probe], labels[gallery], matching)["rank1"]
    accuracy = correctness.mean(0)
    first, second = accuracy[is_eligible], accuracy[~is_eligible]
    rng = np.random.default_rng(seed)
    bootstrap = [rng.choice(first, len(first), replace=True).mean() - rng.choice(second, len(second), replace=True).mean()
                 for _ in range(resamples)]
    # Fisher is meaningful for one binary probe decision per identity; report it
    # on the first fixed schedule, separately from the across-schedule averages.
    binary = correctness[0]
    result = {"eligible_rank1": float(first.mean()), "other_rank1": float(second.mean()),
              "difference": float(first.mean() - second.mean()),
              "ci95_identity_bootstrap": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "eligible_identities": len(first), "other_identities": len(second)}
    if np.isin(binary, [0, 1]).all():
        a, b = int(binary[is_eligible].sum()), int(binary[~is_eligible].sum())
        result["fisher_two_sided_p_first_schedule"] = float(fisher_exact([[a, len(first) - a], [b, len(second) - b]]).pvalue)
    return result
