"""Training and evaluation protocols for experiments E1--E7."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .config import seed_everything
from .data import write_json
from .baselines import make_protector
from .training.alternate_trainer import AlternateTrainer, load_protector
from .training.auditor_trainer import AuditorTrainer
from .protocols.history_episode_builder import HistoryEpisodeBuilder
from .evaluation.recognition import evaluate_recognition, candidate_recognition


def obtain_model(store, config, output, device, checkpoint=None):
    if checkpoint:
        model, saved = load_protector(checkpoint, device)
        if saved["config"]["dataset"] != store.dataset:
            raise ValueError("The protector checkpoint belongs to another dataset")
        return model
    method = config.get("method", "cegait")
    model = make_protector(method, **config.get("model", {})).to(device)
    if not any(p.requires_grad for p in model.parameters()):
        return model.eval()
    seed_everything(config.get("seed", 42), config.get("threads", 4))
    return AlternateTrainer(store, config, output, device).run()


def audit_case(model, store, config, output, device, family=None, lengths=(1, 2, 4),
               evaluate=(4,), candidate="nm-06@090", kind="query", knowledge="secret", cohort_length=4):
    family = family or ("cond" if store.dataset == "casia_b" else "seq")
    seed_everything(config.get("seed", 42), config.get("threads", 4))
    auditor = AuditorTrainer(model, store, config, output, device, kind, family,
                             lengths, candidate, knowledge, cohort_length).fit()
    return {str(length): auditor.evaluate(length) for length in evaluate}


def run_experiment(name, store, config, output, device="cpu", checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": name, "dataset": store.dataset, "cases": {}}
    write_json(output / "config.json", config)

    def trained_case(label, override=None):
        current = deepcopy(config)
        if override:
            for key, values in override.items():
                if isinstance(values, dict):
                    current.setdefault(key, {}).update(values)
                else:
                    current[key] = values
        model = obtain_model(store, current, output / label / "protector", device)
        recognition = evaluate_recognition(model, store, schedules=current["auditing"]["schedules"], device=device)
        audits = audit_case(model, store, current, output / label / "audit", device)
        summary["cases"][label] = {"recognition": recognition, "linkage": audits,
                                    "parameters": sum(p.numel() for p in model.parameters())}
        write_json(output / "summary.json", summary)

    if name == "e1":
        cases = [("cegait", {}), ("late_key_cm", {"method": "late_key_cm"}),
                 ("mlp_hash_cm", {"method": "mlp_hash_cm"}),
                 ("without_mlg", {"training": {"beta": 0.0}}),
                 ("mean_group", {"training": {"mean_groups": True}}),
                 ("pairwise_only", {"protocol": {"lengths": [1]}}),
                 ("without_local_remap", {"model": {"local_remap": False}}),
                 ("without_key_weighting", {"model": {"key_weighting": False}})]
        for label, changes in cases:
            trained_case(label, changes)
    elif name == "e2":
        methods = ["unprotected", "cegait", "priv_ihgr", "biohash", "iom_grp", "iom_urp", "rand_hash", "mlp_hash", "simple_repair"]
        for method in methods:
            changes = {"method": method}
            if method == "priv_ihgr":
                changes["training"] = {"lambda_e": 0.0}
            trained_case(method, changes)
    else:
        if name in {"e3", "e4", "e5", "e6"}:
            model = obtain_model(store, config, output / "protector", device, checkpoint)
        if name == "e3":
            candidates = ["nm-06@090", "nm-06@054", "bg-02@090", "cl-02@090"]
            if store.dataset != "casia_b":
                raise ValueError("E3 uses the controlled CASIA-B candidate configurations")
            for candidate in candidates:
                summary["cases"][candidate] = audit_case(model, store, config, output / candidate.replace("@", "_"), device, candidate=candidate)
                summary["cases"][candidate]["recognition"] = candidate_recognition(model, store, candidate,
                    schedules=config["auditing"]["schedules"], device=device)
        elif name == "e4":
            for family in ["rep", "cond" if store.dataset == "casia_b" else "seq"]:
                summary["cases"][family] = audit_case(model, store, config, output / family,
                                                       device, family, (1, 2, 4, 8), (4, 8), cohort_length=8)
            family = "renewal" if store.dataset == "casia_b" else "seq"
            summary["cases"]["renewal"] = audit_case(model, store, config, output / "renewal", device,
                                                      family, (1, 2, 4, 6, 8), (4, 6, 8), cohort_length=8)
        elif name == "e5":
            for kind in ("deepsets", "transformer"):
                conditions = ("secret", "disclosed", "disclosure_control", "auxiliary", "auxiliary_control")
                for knowledge in conditions:
                    label = f"{kind}_{knowledge}"
                    current = deepcopy(config)
                    current["auditing"].update(steps=20000, batch_size=64)
                    summary["cases"][label] = audit_case(model, store, current, output / label, device,
                                                          kind=kind, knowledge=knowledge)
        elif name == "e6":
            builder = HistoryEpisodeBuilder(store, config.get("seed", 42))
            family = "seq" if store.dataset == "gait3d" else "cond"
            population = len(store.identities("test"))
            coverage = {}
            for length in (1, 2, 4, 8):
                eligible = builder.eligible("test", family, length, cohort_length=length)
                rows = [r for r in store.records if str(r["identity"]) in eligible]
                counts = [len({r.get("tracklet", r["id"]) for r in rows if str(r["identity"]) == identity and r.get("frames", 0) >= 30}) for identity in eligible]
                coverage[str(length)] = {"eligible_identities": len(eligible), "test_identities": population,
                                         "coverage": len(eligible) / population, "median_tracklets": float(np.median(counts)) if counts else 0}
            summary["coverage"] = coverage
            summary["recognition"] = evaluate_recognition(model, store, schedules=config["auditing"]["schedules"], device=device)
            if store.dataset == "gait3d":
                from .evaluation.cohort_coverage import subgroup_recognition
                summary["subgroup_recognition"] = subgroup_recognition(model, store,
                    schedules=config["auditing"]["schedules"], device=device, resamples=10000)
            for length in (1, 2):
                for cohort in (length, 4):
                    label = f"m{length}_cohort{cohort}"
                    summary["cases"][label] = audit_case(model, store, config, output / label,
                                                          device, family, evaluate=(length,), cohort_length=cohort)
        elif name == "e7":
            variations = [("contexts", [2, 4, 8]), ("beta", [0.0, 0.5, 1.0, 2.0]),
                          ("tau", [0.05, 0.1, 0.2, 0.5]), ("lambda_e", [0.0, 0.1, 0.25, 0.5, 1.0]),
                          ("attacker_steps", [1, 2, 3]), ("signed_gain", [True]), ("mean_groups", [True])]
            for parameter, values in variations:
                for value in values:
                    trained_case(f"{parameter}_{value}", {"training": {parameter: value}})
        else:
            raise ValueError(f"Unknown experiment: {name}")
    write_json(output / "summary.json", summary)
    print(f"{name.upper()} completed: {output.resolve()}", flush=True)
    return summary
