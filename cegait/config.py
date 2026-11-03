from pathlib import Path
import random
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path)
    if not path.exists():
        path = ROOT / path
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    return config


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(value="auto"):
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
