"""Identity-cluster / schedule BCa bootstrap, manuscript §4.1.5."""
import numpy as np
from scipy.stats import norm
from .linkage_auc import roc_auc


def mean_schedule_auc(labels, scores, schedules, weights=None, schedule_weights=None):
    unique = np.unique(schedules)
    values = []
    for schedule in unique:
        selected = schedules == schedule
        values.append(roc_auc(labels[selected], scores[selected], None if weights is None else weights[selected]))
    return float(np.average(values, weights=schedule_weights))


def clustered_bca(labels, scores, identities, schedules, resamples=1000, seed=42, paired_scores=None):
    labels, scores = np.asarray(labels), np.asarray(scores)
    identities, schedules = np.asarray(identities), np.asarray(schedules)
    accounts, account_index = np.unique(identities, return_inverse=True)
    schedule_ids = np.unique(schedules)
    if len(accounts) < 3 or resamples < 10:
        raise ValueError("BCa requires >=3 account clusters and >=10 resamples")
    if paired_scores is not None:
        paired_scores = np.asarray(paired_scores)
        if paired_scores.shape != scores.shape:
            raise ValueError("Paired arrays must have identical aligned episodes")

    def statistic(weights=None, schedule_weights=None):
        result = mean_schedule_auc(labels, scores, schedules, weights, schedule_weights)
        if paired_scores is not None:
            result -= mean_schedule_auc(labels, paired_scores, schedules, weights, schedule_weights)
        return result

    estimate = statistic()
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(len(accounts), size=len(accounts)), minlength=len(accounts))
        schedule_counts = np.bincount(rng.integers(len(schedule_ids), size=len(schedule_ids)), minlength=len(schedule_ids))
        replicates.append(statistic(counts[account_index], schedule_counts))
    replicates = np.asarray(replicates)
    jackknife = []
    for index in range(len(accounts)):
        jackknife.append(statistic((account_index != index).astype(float)))
    influence = np.mean(jackknife) - np.asarray(jackknife)
    denominator = 6 * np.sum(influence ** 2) ** 1.5
    acceleration = float(np.sum(influence ** 3) / denominator) if denominator else 0.0
    fraction = (np.sum(replicates < estimate) + 0.5 * np.sum(replicates == estimate)) / len(replicates)
    z0 = norm.ppf(np.clip(fraction, 0.5 / resamples, 1 - 0.5 / resamples))
    z = norm.ppf([0.025, 0.975])
    adjusted = norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
    interval = np.quantile(replicates, np.clip(adjusted, 0, 1))
    return {"estimate": estimate, "ci95_bca": interval.tolist(), "identity_clusters": len(accounts),
            "key_schedules": len(schedule_ids), "bootstrap_resamples": resamples, "acceleration": acceleration}
