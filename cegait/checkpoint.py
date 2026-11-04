"""Checkpoint loading and model-state inspection."""
import hashlib
from pathlib import Path

import torch


def read_checkpoint(path, device="cpu", kind=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    saved = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise ValueError("Expected a version-1 checkpoint mapping")
    if kind is not None and saved.get("kind") != kind:
        raise ValueError(f"Expected a {kind} checkpoint")
    return saved


def checkpoint_summary(path):
    from .baselines import make_protector
    from .models.query_set_adversary import QuerySetAdversary

    path = Path(path)
    saved = read_checkpoint(path, "cpu", "protector")
    model = make_protector(saved["method"], **saved["model_options"])
    model.load_state_dict(saved["protector"], strict=True)
    adversary = QuerySetAdversary()
    adversary.load_state_dict(saved["adversary"], strict=True)
    for component in ("protector", "adversary"):
        if not all(torch.isfinite(value).all().item() for value in saved[component].values()):
            raise ValueError(f"Nonfinite values in {component} state")
    counters = {}
    for field, label in (("gopt", "protector"), ("aopt", "adversary")):
        counters[label] = sorted({int(entry["step"]) for entry in saved[field]["state"].values()})
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    config = saved["config"]
    return {
        "file": path.name,
        "schema_version": saved["schema_version"],
        "kind": saved["kind"],
        "method": saved["method"],
        "dataset": config["dataset"],
        "step": saved["step"],
        "configured_steps": config["training"]["steps"],
        "warmup_steps": config["training"]["warmup_steps"],
        "attacker_steps_per_protector": config["training"]["attacker_steps"],
        "best_validation_recorded": saved["best_validation"],
        "parameters": {
            "protector": sum(parameter.numel() for parameter in model.parameters()),
            "adversary": sum(parameter.numel() for parameter in adversary.parameters()),
        },
        "optimizer_steps": counters,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }
