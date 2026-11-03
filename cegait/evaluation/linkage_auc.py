"""Metrics use the fixed development score direction; never flip on test labels."""
import numpy as np


def roc_auc(labels, scores, weights=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Finite scores and binary labels of equal length required")
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(scores, kind="stable")
    labels, scores, weights = labels[order], scores[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positive = np.add.reduceat(weights * (labels == 1), starts)
    negative = np.add.reduceat(weights * (labels == 0), starts)
    denominator = positive.sum() * negative.sum()
    if denominator <= 0:
        raise ValueError("AUC requires both labels with positive weight")
    return float(np.sum(positive * (np.cumsum(negative) - 0.5 * negative)) / denominator)


def calibrate_threshold(negative_scores, fpr=0.01):
    scores = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if not len(scores) or not np.isfinite(scores).all() or not 0 < fpr < 1:
        raise ValueError("Finite non-mated calibration scores and 0<fpr<1 required")
    allowed = min(int(np.floor(len(scores) * fpr)), len(scores) - 1)
    return float(np.nextafter(scores[allowed], np.inf))


def operating_point(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    return {"tpr_at_nominal_1pct_fpr": float((scores[labels == 1] >= threshold).mean()),
            "realized_test_fpr": float((scores[labels == 0] >= threshold).mean()), "threshold": threshold}


def system_linkability(labels, scores, bins=40):
    labels, scores = np.asarray(labels), np.asarray(scores)
    lower, upper = float(scores.min()), float(scores.max())
    if lower == upper:
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    p1 = np.histogram(scores[labels == 1], edges)[0].astype(float)
    p0 = np.histogram(scores[labels == 0], edges)[0].astype(float)
    p1 /= p1.sum()
    p0 /= p0.sum()
    local = np.divide(p1 - p0, p1 + p0, out=np.zeros_like(p1), where=(p1 + p0) > 0).clip(0)
    return float(np.sum(p1 * local))
