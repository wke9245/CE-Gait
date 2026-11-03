import numpy as np
import torch
import torch.nn.functional as F
from ..protocols.domain_key_schedule import DomainKeySchedule


def rank_metrics(probe, gallery, probe_ids, gallery_ids, matching="cosine"):
    probe, gallery = np.asarray(probe), np.asarray(gallery)
    probe_ids, gallery_ids = np.asarray(probe_ids), np.asarray(gallery_ids)
    if not len(probe) or not len(gallery):
        raise ValueError("Nonempty gallery and probe required")
    if matching in ("collision", "hamming"):
        distance = np.stack([(p[None] != gallery).mean(-1) for p in probe])
    else:
        p = probe / np.linalg.norm(probe, axis=-1, keepdims=True).clip(1e-12)
        g = gallery / np.linalg.norm(gallery, axis=-1, keepdims=True).clip(1e-12)
        distance = (1 - p @ g.T) / 2
    order = np.argsort(distance, axis=1, kind="stable")
    matches = gallery_ids[order] == probe_ids[:, None]
    valid = matches.any(1)
    if not valid.any():
        raise ValueError("No probe identity has a gallery observation")
    precision = matches.cumsum(1) / np.arange(1, len(gallery) + 1)
    average_precision = (precision * matches).sum(1) / matches.sum(1).clip(1)
    return {"rank1": float(matches[valid, 0].mean()), "mAP": float(average_precision[valid].mean()),
            "probes": int(valid.sum())}


@torch.no_grad()
def candidate_recognition(model, store, candidate="nm-06@090", schedules=10, seed=42, device="cpu"):
    sequence, angle = candidate.split("@")
    rows = [r for r in store.records if r["role"] == "test"]
    indices = [r["index"] for r in rows]
    labels = np.array([str(r["identity"]) for r in rows])
    views = np.array([str(r["view"]).zfill(3) for r in rows])
    sequences = np.array([r["sequence"] for r in rows])
    p = (sequences == sequence) & (views == angle)
    matching = getattr(model, "matching", "cosine")
    accuracies = []
    model.eval()
    for schedule in range(schedules):
        key = DomainKeySchedule(seed, "recognition/test").vectors(schedule, ["active"], device)
        templates = []
        for start in range(0, len(rows), 128):
            result = model(store.tensor(indices[start:start + 128], device, getattr(model, "feature_level", False)), key)
            templates.append(result["codes" if matching != "cosine" else "template"].cpu().numpy())
        templates = np.concatenate(templates)
        for view in np.unique(views):
            g = np.isin(sequences, ["nm-01", "nm-02", "nm-03", "nm-04"]) & (views == view)
            if view != angle and p.any() and g.any():
                accuracies.append(rank_metrics(templates[p], templates[g], labels[p], labels[g], matching)["rank1"])
    return {"candidate": candidate, "rank1": float(np.mean(accuracies)), "gallery_view_comparisons": len(accuracies)}


@torch.no_grad()
def evaluate_recognition(model, store, role="test", schedules=10, seed=42, device="cpu", batch_size=128):
    model.eval()
    indices = [r["index"] for r in store.records if r["role"] == role]
    rows = [store.records[i] for i in indices]
    results = []
    matching = getattr(model, "matching", "cosine")
    for schedule in range(schedules):
        key = DomainKeySchedule(seed, f"recognition/{role}").vectors(schedule, ["active"], device)
        encoded = []
        for start in range(0, len(indices), batch_size):
            output = model(store.tensor(indices[start:start + batch_size], device, getattr(model, "feature_level", False)), key)
            encoded.append(output["codes" if matching != "cosine" else "template"].cpu().numpy())
        templates = np.concatenate(encoded)
        ids = np.array([r["identity"] for r in rows])
        if store.dataset == "gait3d":
            gallery = np.array([r["gallery"] for r in rows])
            results.append(rank_metrics(templates[~gallery], templates[gallery], ids[~gallery], ids[gallery], matching))
            continue
        seq = np.array([r["sequence"] for r in rows])
        views = np.array([str(r["view"]).zfill(3) for r in rows])
        gallery = np.isin(seq, ["nm-01", "nm-02", "nm-03", "nm-04"])
        conditions = {}
        for condition, sequences in {"NM": ["nm-05", "nm-06"], "BG": ["bg-01", "bg-02"], "CL": ["cl-01", "cl-02"]}.items():
            view_results = []
            for probe_view in np.unique(views):
                p = np.isin(seq, sequences) & (views == probe_view)
                for gallery_view in np.unique(views):
                    if probe_view == gallery_view:
                        continue
                    g = gallery & (views == gallery_view)
                    if p.any() and g.any():
                        view_results.append(rank_metrics(templates[p], templates[g], ids[p], ids[g], matching)["rank1"])
            if view_results:
                conditions[condition] = float(np.mean(view_results))
        results.append({**conditions, "mean_rank1": float(np.mean(list(conditions.values())))})
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}
