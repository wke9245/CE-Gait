"""Identity-supervised IHGR pretraining; frozen feature extraction afterwards."""
from pathlib import Path
import numpy as np
import torch
from ..backbone import IHGRNet, IHGRAdapter
from ..data import read_json, write_json, load_sequence


def sequence_path(manifest_path, record):
    path = Path(record["path"])
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def pretrain_backbone(manifest_path, output, steps=80000, device="cpu", seed=42,
                      identities_per_batch=8, clips_per_identity=4, resume=None):
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "silhouettes" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a version-1 silhouette manifest produced by the prepare command")
    rows = [r for r in manifest["records"] if r["role"] != "test"]
    identities = sorted({str(r["identity"]) for r in rows})
    if len(identities) < 2:
        raise ValueError("IHGR pretraining needs >=2 development identities")
    by_id = {identity: [r for r in rows if str(r["identity"]) == identity] for identity in identities}
    mapping = {identity: i for i, identity in enumerate(identities)}
    model = IHGRNet(len(identities)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(seed)
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["identity_mapping"] != mapping:
            raise ValueError("Checkpoint identity mapping differs from this manifest")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start, steps):
        selected = rng.choice(identities, min(identities_per_batch, len(identities)), replace=False)
        images, labels = [], []
        for identity in selected:
            pool = by_id[identity]
            chosen = rng.choice(len(pool), clips_per_identity, replace=len(pool) < clips_per_identity)
            for index in chosen:
                image, _ = load_sequence(sequence_path(manifest_path, pool[index]), training=True, rng=rng)
                images.append(image)
                labels.append(mapping[identity])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(torch.stack(images).to(device))
        loss = model.training_loss(result, torch.tensor(labels, device=device), 0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            torch.save({"schema_version": 1, "kind": "ihgr", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step + 1, "num_classes": len(identities), "identity_mapping": mapping,
                        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "architecture": "IHGR-Net",
                        "sparsity_weight": 0.5, "seed": seed}, output)
            print(f"IHGR {step + 1}/{steps}, loss={float(loss.detach()):.4f}", flush=True)
    return output


@torch.no_grad()
def extract_features(manifest_path, checkpoint, output, device="cpu"):
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "silhouettes" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a version-1 silhouette manifest produced by the prepare command")
    adapter = IHGRAdapter.from_checkpoint(checkpoint, device).eval()
    features, embeddings, rows = [], [], []
    for index, record in enumerate(manifest["records"]):
        images, usable = load_sequence(sequence_path(manifest_path, record))
        result = adapter.network(images.unsqueeze(0).to(device))
        features.append((result["local"] if isinstance(result, dict) else result)[0].cpu().numpy())
        embeddings.append((result["embedding"] if isinstance(result, dict) else result.mean(1))[0].cpu().numpy())
        rows.append({**record, "frames": usable})
        if (index + 1) % 100 == 0:
            print(f"Extracted {index + 1}/{len(manifest['records'])}", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_name = output.stem + "_features.npz"
    np.savez_compressed(output.parent / feature_name, local=np.stack(features), pooled=np.stack(embeddings))
    write_json(output, {"schema_version": 1, "kind": "features", "dataset": manifest["dataset"], "features": feature_name,
                        "backbone_checkpoint": str(Path(checkpoint).resolve()),
                        "identity_roles": manifest["identity_roles"], "records": rows})
    return output
