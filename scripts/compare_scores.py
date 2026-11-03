"""Paired differences with matched identity/schedule bootstrap draws."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cegait.evaluation.bootstrap_bca import clustered_bca


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--output", default="paired_difference.json")
    args = parser.parse_args()
    first, second = read(args.first), read(args.second)
    columns = ["schedule", "identity", "episode", "label"]
    if [[r[k] for k in columns] for r in first] != [[r[k] for k in columns] for r in second]:
        raise ValueError("Paired comparison requires identical aligned episode/account/schedule identifiers")
    result = clustered_bca(np.array([int(r["label"]) for r in first]),
                           np.array([float(r["score"]) for r in first]),
                           np.array([r["identity"] for r in first]),
                           np.array([r["schedule"] for r in first]),
                           resamples=args.resamples,
                           paired_scores=np.array([float(r["score"]) for r in second]))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
