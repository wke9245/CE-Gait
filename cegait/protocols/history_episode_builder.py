"""Observation-level episodes shared by methods and key schedules."""
from dataclasses import dataclass
import numpy as np
import torch
from .domain_key_schedule import DomainKeySchedule


@dataclass
class EpisodeBatch:
    history: np.ndarray
    candidate: np.ndarray
    labels: np.ndarray
    identities: list
    draws: np.ndarray

    def __len__(self):
        return len(self.labels)

    def subset(self, indices):
        return EpisodeBatch(self.history[indices], self.candidate[indices], self.labels[indices],
                            [self.identities[i] for i in indices], self.draws[indices])

    def as_dict(self):
        return {"history": self.history.tolist(), "candidate": self.candidate.tolist(),
                "labels": self.labels.tolist(), "identities": self.identities, "draws": self.draws.tolist()}


class HistoryEpisodeBuilder:
    def __init__(self, store, seed=42):
        self.store, self.seed = store, seed

    def observations(self, identity, family, length, candidate="nm-06@090", cohort_length=4, renewal=False):
        rows = self.store.records_for(identity)
        if self.store.dataset == "gait3d":
            unique = {r.get("tracklet", r["id"]): r for r in rows if r.get("frames", 0) >= 30}
            rows = sorted(unique.values(), key=lambda r: r["id"])
            required = max(length, cohort_length) + (2 if renewal else 1)
            if len(rows) < required:
                return None
            historical = [rows[0]] * length if family == "rep" else rows[:length]
            return [r["index"] for r in historical], rows[required - 1]["index"]
        by_config = {(r["sequence"], str(r["view"]).zfill(3)): r["index"] for r in rows}
        seq, view = candidate.split("@")
        if (seq, view.zfill(3)) not in by_config:
            return None
        if family == "rep":
            configs = [("nm-01", "090")] * length
        elif family == "seq":
            if length > 4:
                return None
            configs = [(f"nm-{i:02}", "090") for i in range(1, length + 1)]
        elif family == "view":
            angles = {1: [36], 2: [36, 108], 4: [36, 72, 108, 144]}.get(length)
            if angles is None:
                return None
            configs = [("nm-01", f"{angle:03}") for angle in angles]
        elif family == "cond":
            order = ["nm-01", "bg-01", "nm-02", "cl-01", "nm-03", "nm-04", "bg-02", "cl-02"]
            configs = [(s, "090") for s in order[:length]]
            if len(configs) != length:
                return None
        elif family == "renewal":
            order = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "bg-02", "cl-01", "cl-02"]
            configs = [(s, "090") for s in order[:length]]
        else:
            raise ValueError(f"Unsupported exposure family: {family}")
        if any(config not in by_config for config in configs):
            return None
        history, query = [by_config[c] for c in configs], by_config[(seq, view.zfill(3))]
        if query in history:
            return None
        return history, query

    def eligible(self, role, family, length, candidate="nm-06@090", cohort_length=4, renewal=False):
        result = {}
        for identity in self.store.identities(role):
            observations = self.observations(identity, family, length, candidate, cohort_length, renewal)
            if observations is not None:
                result[identity] = observations
        return result

    def build(self, role, family="cond", length=4, draws=20, candidate="nm-06@090",
              cohort_length=4, renewal=False):
        eligible = self.eligible(role, family, length, candidate, cohort_length, renewal)
        pool = sorted(eligible)
        if len(pool) < 2:
            raise ValueError(f"Fewer than two eligible identities: {role}/{family}/m={length}")
        rng = np.random.default_rng(self.seed)
        history, queries, labels, accounts, draw_ids = [], [], [], [], []
        for draw in range(draws):
            # Cyclic permutation without fixed points preserves candidate marginals.
            permutation = list(rng.permutation(pool))
            shift = int(rng.integers(1, len(pool)))
            negatives = dict(zip(permutation, permutation[shift:] + permutation[:shift]))
            for identity in pool:
                h, positive = eligible[identity]
                negative = eligible[negatives[identity]][1]
                for label, query in ((1, positive), (0, negative)):
                    history.append(h)
                    queries.append(query)
                    labels.append(label)
                    accounts.append(identity)
                    draw_ids.append(draw)
        return EpisodeBatch(np.asarray(history), np.asarray(queries), np.asarray(labels, dtype=np.float32),
                            accounts, np.asarray(draw_ids))


def encode_episodes(model, store, episodes, schedule, namespace, seed=42, device="cpu", disclose=False):
    """Domain tuple depends on schedule/draw, and never on account or label."""
    n, m = episodes.history.shape
    keys = DomainKeySchedule(seed, namespace)
    hkeys = torch.stack([keys.vectors(f"{schedule}/{draw}", range(m), device) for draw in episodes.draws])
    ckeys = torch.stack([keys.vectors(f"{schedule}/{draw}", ["candidate"], device)[0] for draw in episodes.draws])
    feature_level = getattr(model, "feature_level", False)
    inputs = torch.cat([store.tensor(episodes.history.reshape(-1), device, feature_level),
                        store.tensor(episodes.candidate, device, feature_level)])
    vectors = torch.cat([hkeys.reshape(-1, 256), ckeys])
    output = model(inputs, vectors)
    h = output["template"][:n * m].reshape(n, m, -1)
    c = output["template"][n * m:]
    slots = hkeys.new_zeros(n, m, 257)
    if disclose:
        revealed = min(m, 2)
        slots[:, :revealed, :256] = hkeys[:, :revealed]
        slots[:, :revealed, 256] = 1
    return h, c, torch.as_tensor(episodes.labels, device=device), output, slots
