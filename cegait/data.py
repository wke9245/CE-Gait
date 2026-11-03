"""Versioned sequence/feature manifests, compatible with OpenGait layout."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def load_sequence(path, frames=30, training=False, rng=None):
    path = Path(path)
    if path.is_dir():
        from PIL import Image
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp"))
        x = np.stack([np.array(Image.open(p).convert("L")) for p in files])
    elif path.suffix == ".pkl":
        with path.open("rb") as handle:
            x = np.asarray(pickle.load(handle))
    else:
        x = np.load(path, allow_pickle=False)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"Expected [frames,height,width] silhouettes: {path}")
    x = x[np.any(x > 0, axis=(1, 2))]
    usable = len(x)
    if not usable:
        raise ValueError(f"Empty silhouette sequence: {path}")
    if training and usable >= frames:
        rng = np.random.default_rng() if rng is None else rng
        indices = np.sort(rng.choice(usable, frames, replace=False))
    else:
        indices = np.linspace(0, usable - 1, frames).astype(int)
    x = torch.from_numpy((x[indices] > 0).astype(np.float32)).unsqueeze(1)
    return F.interpolate(x, (64, 44), mode="nearest"), usable


def assign_roles(identities, dataset, partition=None):
    if dataset not in {"casia_b", "gait3d"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "casia_b":
        def role(identity):
            value = int(identity)
            if not 1 <= value <= 124:
                raise ValueError("CASIA-B identity numbers must be in the range 001--124")
            return "generator" if value <= 30 else "auditor" if value <= 60 else "validation" if value <= 68 else "calibration" if value <= 74 else "test"
        return {identity: role(identity) for identity in identities}
    if partition is None:
        raise ValueError("Gait3D requires the official partition JSON")
    train = sorted(str(i) for i in partition["TRAIN_SET"])
    test = set(str(i) for i in partition["TEST_SET"])
    if len(train) != 3000 or len(test) != 1000 or set(train) & test:
        raise ValueError("Gait3D requires the disjoint official 3000/1000 identity partition")
    development = {identity: "generator" if i < 2500 else "validation" if i < 2800 else "calibration"
                   for i, identity in enumerate(train)}
    result = {i: "test" if i in test else development.get(i) for i in identities}
    if any(r is None for r in result.values()):
        raise ValueError("An identity is absent from the official partition")
    return result


def build_sequence_manifest(root, dataset, output, partition=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Silhouette directory does not exist: {root}")
    files = sorted(root.rglob("*.pkl")) + sorted(root.rglob("*.npy"))
    if not files:
        # Also accept identity/sequence/view directories of PNG silhouettes.
        files = sorted(set(p.parent for p in root.rglob("*.png")))
    split = read_json(partition) if partition else None
    if dataset == "gait3d" and (split is None or not split.get("PROBE_SET")):
        raise ValueError("Gait3D indexing requires official identity and probe assignments")
    rows = []
    for path in files:
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        identity, sequence, view = relative.parts[:3]
        if path.is_file() and len(relative.parts) == 3:
            view = path.stem
        tracklet = f"{identity}-{sequence}-{view}"
        rows.append({"id": tracklet, "identity": identity, "sequence": sequence.lower(),
                     "view": view, "path": str(path), "tracklet": tracklet,
                     "gallery": bool(dataset == "gait3d" and split and tracklet not in split.get("PROBE_SET", []))})
    if not rows:
        raise ValueError("No silhouette sequences found in identity/sequence/view layout")
    roles = assign_roles(sorted({r["identity"] for r in rows}), dataset, split)
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate trajectory (choose a single silhouette modality): {row['id']}")
        seen.add(row["id"])
        row["role"] = roles[row["identity"]]
    write_json(output, {"schema_version": 1, "dataset": dataset, "kind": "silhouettes",
                        "identity_roles": roles, "records": rows})
    return output


class FeatureStore:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.manifest = read_json(self.path)
        if self.manifest.get("schema_version") != 1 or self.manifest.get("kind") != "features":
            raise ValueError("Expected a version-1 feature manifest produced by the extract command")
        self.dataset = self.manifest["dataset"]
        if self.dataset not in {"casia_b", "gait3d"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        feature_file = Path(self.manifest["features"])
        if not feature_file.is_absolute():
            feature_file = self.path.parent / feature_file
        archive = np.load(feature_file, allow_pickle=False)
        self.features = archive["local"].astype(np.float32)
        self.pooled = archive["pooled"].astype(np.float32)
        archive.close()
        self.records = self.manifest["records"]
        if self.features.shape != (len(self.records), 16, 256) or not np.isfinite(self.features).all():
            raise ValueError("Finite feature array [records,16,256] required")
        if self.pooled.shape != (len(self.records), 256) or not np.isfinite(self.pooled).all():
            raise ValueError("Finite pooled feature array [records,256] required")
        if not self.records:
            raise ValueError("The feature manifest contains no observations")
        expected_roles = self.manifest["identity_roles"]
        allowed_roles = {"generator", "auditor", "validation", "calibration", "test"}
        roles, identifiers = {}, set()
        self.by_identity = {}
        for index, row in enumerate(self.records):
            row["index"] = index
            identity, role = str(row["identity"]), row["role"]
            if role not in allowed_roles or expected_roles.get(identity) != role:
                raise ValueError(f"Invalid identity-role assignment: {identity}")
            if self.dataset == "casia_b" and assign_roles([identity], "casia_b")[identity] != role:
                raise ValueError(f"CASIA-B identity does not match the prescribed partition: {identity}")
            if not isinstance(row.get("frames"), int) or row["frames"] < 1:
                raise ValueError("Each observation must record its original nonempty frame count")
            if identity in roles and roles[identity] != role:
                raise ValueError("Identity leakage across development/test roles")
            if row["id"] in identifiers:
                raise ValueError("Duplicate sequence identifier")
            roles[identity] = role
            identifiers.add(row["id"])
            self.by_identity.setdefault(identity, []).append(row)
        self.roles = roles

    def identities(self, role):
        return sorted(identity for identity, assigned in self.roles.items() if assigned == role)

    def records_for(self, identity):
        return self.by_identity.get(str(identity), [])

    def tensor(self, indices, device="cpu", pooled=False):
        array = self.pooled if pooled else self.features
        return torch.as_tensor(array[np.asarray(indices)], device=device)

    def utility_batch(self, role, identities, clips, rng, device="cpu", pooled=False):
        pool = self.identities(role)
        if len(pool) < 2:
            raise ValueError("At least two development identities required")
        chosen = rng.choice(pool, min(identities, len(pool)), replace=False)
        indices, labels = [], []
        for label, identity in enumerate(chosen):
            rows = self.records_for(identity)
            if len(rows) < 2:
                raise ValueError("Each utility identity needs two independent observations")
            selected = rng.choice(len(rows), clips, replace=len(rows) < clips)
            indices.extend(rows[i]["index"] for i in selected)
            labels.extend([label] * clips)
        return self.tensor(indices, device, pooled), torch.tensor(labels, device=device)
