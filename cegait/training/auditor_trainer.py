import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..models.independent_auditors import make_auditor
from ..protocols.history_episode_builder import HistoryEpisodeBuilder, EpisodeBatch, encode_episodes
from ..evaluation.linkage_auc import roc_auc, calibrate_threshold, operating_point, system_linkability
from ..evaluation.bootstrap_bca import clustered_bca
from ..data import write_json


def auxiliary_pairs(plan, store):
    """Same-observation cross-key supervision, training only, balanced controls."""
    history = plan.history[:, :1].copy()
    candidates = []
    for hindex, cindex in zip(history[:, 0], plan.candidate):
        hrecord, crecord = store.records[int(hindex)], store.records[int(cindex)]
        pool = store.records_for(crecord["identity"])
        if store.dataset == "casia_b":
            pool = [r for r in pool if r["sequence"] == hrecord["sequence"] and r["view"] == hrecord["view"]]
        else:
            pool = [r for r in pool if r.get("frames", 0) >= 30]
        candidates.append(sorted(pool, key=lambda r: r["id"])[0]["index"])
    candidates = np.asarray(candidates)
    return EpisodeBatch(history, candidates, plan.labels.copy(), list(plan.identities), plan.draws.copy())


class AuditorTrainer:
    def __init__(self, model, store, config, output, device="cpu", kind="query", family="cond",
                 lengths=(1, 2, 4), candidate="nm-06@090", knowledge="secret", cohort_length=4):
        self.model, self.store, self.config = model, store, config
        self.output, self.device = Path(output), device
        self.output.mkdir(parents=True, exist_ok=True)
        self.kind, self.family, self.lengths, self.candidate = kind, family, list(lengths), candidate
        self.knowledge, self.cohort_length = knowledge, cohort_length
        self.seed = config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed + 1001)
        self.builder = HistoryEpisodeBuilder(store, self.seed)
        self.model.eval().requires_grad_(False)
        dimension = 1024 if getattr(model, "kind", "") in ("biohash", "rand_hash") else 256
        self.auditor = make_auditor(kind, disclose_keys=knowledge in ("disclosed", "disclosure_control"), dimension=dimension).to(device)
        self.options = config["auditing"]
        self.optimizer = torch.optim.AdamW(self.auditor.parameters(), lr=self.options["lr"], weight_decay=1e-4)
        role = "auditor" if store.dataset == "casia_b" else "generator"
        self.train_plans = {m: self.builder.build(role, family, m, draws=self.options.get("training_draws", 20),
                                                candidate=candidate, cohort_length=max(lengths)) for m in lengths}
        original_singletons = self.train_plans[min(lengths)]
        budget = self.options.get("training_episodes", 4000)
        for i, length in enumerate(lengths):
            plan = self.train_plans[length]
            count = budget // (2 * len(lengths)) + (i < (budget // 2) % len(lengths))
            chosen = self.rng.choice(len(plan) // 2, count, replace=count > len(plan) // 2) * 2
            self.train_plans[length] = plan.subset(np.stack([chosen, chosen + 1], 1).reshape(-1))
        self.extra_plan = None
        if knowledge in ("auxiliary", "auxiliary_control"):
            if min(lengths) != 1:
                raise ValueError("Auxiliary controls require singleton episodes in training")
            extra_pairs = self.options.get("auxiliary_pairs", 500 if store.dataset == "casia_b" else 1000)
            selected = self.rng.choice(len(original_singletons) // 2, extra_pairs,
                                       replace=extra_pairs > len(original_singletons) // 2) * 2
            extra = original_singletons.subset(np.stack([selected, selected + 1], 1).reshape(-1))
            self.extra_plan = auxiliary_pairs(extra, store) if knowledge == "auxiliary" else extra
        write_json(self.output / "training_episodes.json", {
            "ordinary": {str(m): plan.as_dict() for m, plan in self.train_plans.items()},
            "extra": self.extra_plan.as_dict() if self.extra_plan is not None else None,
            "ordinary_episodes": sum(len(p) for p in self.train_plans.values()),
            "extra_episodes": len(self.extra_plan) if self.extra_plan is not None else 0})
        self.validation_plans = {m: self.builder.build("validation", family, m, draws=1,
                                                     candidate=candidate, cohort_length=max(lengths)) for m in lengths}

    def predict(self, episodes, schedule, namespace):
        with torch.no_grad():
            h, c, y, _, slots = encode_episodes(self.model, self.store, episodes, schedule, namespace,
                                               self.seed, self.device, self.knowledge == "disclosed")
        if self.knowledge in ("disclosed", "disclosure_control"):
            logits = self.auditor(h, c, key_slots=slots)
        else:
            logits = self.auditor(h, c)
        return logits, y

    @torch.no_grad()
    def validate(self):
        self.auditor.eval()
        aucs = []
        for length, plan in self.validation_plans.items():
            scores, labels = [], []
            for start in range(0, len(plan), self.options["batch_size"]):
                batch = plan.subset(np.arange(start, min(len(plan), start + self.options["batch_size"])))
                logits, y = self.predict(batch, 0, "audit_validation")
                scores.extend(logits.cpu().tolist())
                labels.extend(y.cpu().tolist())
            aucs.append(roc_auc(labels, scores))
        return float(np.mean(aucs))

    def fit(self):
        best, validation = -float("inf"), None
        for step in range(self.options["steps"]):
            self.auditor.train()
            length = self.lengths[step % len(self.lengths)]
            plan = self.train_plans[length]
            size = self.options["batch_size"]
            pairs = self.rng.integers(len(plan) // 2, size=max(1, size // 2)) * 2
            batch = plan.subset(np.stack([pairs, pairs + 1], 1).reshape(-1))
            # Every enhanced update has a budget-matched ordinary-pair control.
            if self.knowledge in ("auxiliary", "auxiliary_control") and step % self.options.get("auxiliary_every", 5) == 0:
                extra = self.extra_plan
                pairs = self.rng.integers(len(extra) // 2, size=max(1, size // 2)) * 2
                batch = extra.subset(np.stack([pairs, pairs + 1], 1).reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            logits, labels = self.predict(batch, step, "audit_training")
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.auditor.parameters(), 5)
            self.optimizer.step()
            if (step + 1) % self.options["validate_every"] == 0 or step + 1 == self.options["steps"]:
                validation = self.validate()
                if validation > best:
                    best = validation
                    torch.save({"auditor": self.auditor.state_dict(), "kind": self.kind,
                                "knowledge": self.knowledge, "step": step + 1, "validation_auc": validation,
                                "family": self.family, "lengths": self.lengths, "candidate": self.candidate,
                                "seed": self.seed}, self.output / "auditor.pt")
        state = torch.load(self.output / "auditor.pt", map_location=self.device, weights_only=False)
        self.auditor.load_state_dict(state["auditor"])
        self.auditor.eval()
        return self

    @torch.no_grad()
    def score_plan(self, plan, schedule, namespace):
        scores = []
        for start in range(0, len(plan), self.options["batch_size"]):
            batch = plan.subset(np.arange(start, min(len(plan), start + self.options["batch_size"])))
            logits, _ = self.predict(batch, schedule, namespace)
            scores.extend(logits.sigmoid().cpu().tolist())
        return np.asarray(scores)

    def evaluate(self, length=4, schedules=None, draws=None):
        schedules = self.options["schedules"] if schedules is None else schedules
        draws = self.options["evaluation_draws"] if draws is None else draws
        plan = self.builder.build("test", self.family, length, draws=draws, candidate=self.candidate,
                                  cohort_length=self.cohort_length)
        calibration = self.builder.build("calibration", self.family, length, draws=draws,
                                         candidate=self.candidate, cohort_length=self.cohort_length)
        prefix = self.output / f"m{length}"
        prefix.mkdir(exist_ok=True)
        write_json(prefix / "episodes.json", {"feature_manifest": str(self.store.path),
                                             "family": self.family, "candidate": self.candidate,
                                             "seed": self.seed, "episodes": plan.as_dict()})
        rows, per_schedule = [], []
        for schedule in range(schedules):
            cal_scores = self.score_plan(calibration, schedule, "audit_calibration")
            threshold = calibrate_threshold(cal_scores[calibration.labels == 0])
            scores = self.score_plan(plan, schedule, "audit_test")
            metrics = {"schedule": schedule, "auc": roc_auc(plan.labels, scores),
                       **operating_point(plan.labels, scores, threshold),
                       "system_linkability": system_linkability(plan.labels, scores)}
            per_schedule.append(metrics)
            for index, score in enumerate(scores):
                rows.append({"schedule": schedule, "identity": plan.identities[index], "episode": index,
                             "draw": int(plan.draws[index]), "label": int(plan.labels[index]), "score": float(score)})
        with (prefix / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = clustered_bca([r["label"] for r in rows], [r["score"] for r in rows],
                                [r["identity"] for r in rows], [r["schedule"] for r in rows],
                                self.options["bootstrap_resamples"], self.seed)
        summary.update({"auc": float(np.mean([s["auc"] for s in per_schedule])),
                        "tpr_at_nominal_1pct_fpr": float(np.mean([s["tpr_at_nominal_1pct_fpr"] for s in per_schedule])),
                        "realized_test_fpr": float(np.mean([s["realized_test_fpr"] for s in per_schedule])),
                        "system_linkability": float(np.mean([s["system_linkability"] for s in per_schedule])),
                        "auditor": self.kind, "knowledge": self.knowledge, "family": self.family,
                        "history_length": length, "per_schedule": per_schedule})
        write_json(prefix / "metrics.json", summary)
        print(f"Audit {self.kind}/{self.family}/m={length}: AUC={summary['auc']:.4f}", flush=True)
        return summary
