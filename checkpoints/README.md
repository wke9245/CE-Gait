# Checkpoint

The CASIA-B CE-Gait checkpoint at protector iteration 30,000 is distributed through [CE-Gait Release v0.1.0](https://github.com/wke9245/CE-Gait/releases/tag/v0.1.0). The release inherits this repository's private visibility. The serialized weights, optimizer states, iteration counters, and RNG states are preserved in full.

The release contains `best.pt`, `best.json`, and `SHA256SUMS`. Release and asset identifiers, download URLs, and SHA-256 digests are recorded in `release.json`.

| Component | Recorded value |
|---|---:|
| Protector parameters | 2,368,257 |
| In-loop adversary parameters | 788,225 |
| Protector optimizer updates | 30,000 |
| Adversary optimizer updates | 40,000 |
| File size | 37,940,917 bytes |

SHA-256: `6fec75c388e3343e276faded78fbb8a26b743c10c3589cf10f01aaebcbe54933`

```bash
python scripts/download_checkpoint.py
python run.py checkpoint-info --checkpoint checkpoints/best.pt
```

The download script reads a GitHub token with repository read access from `GH_TOKEN` or `GITHUB_TOKEN`. A local token file can be selected with `--token-file`. Each file is verified before it replaces the local destination.

The checkpoint includes protection and in-loop attacker states. The frozen IHGR-Net encoder is loaded separately. Detailed metadata is stored in `best.json`.
