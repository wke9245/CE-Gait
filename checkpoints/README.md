# Checkpoint

The CASIA-B CE-Gait checkpoint at protector iteration 30,000 is distributed through the private [ce-gait-checkpoint package](https://github.com/users/wke9245/packages/container/package/ce-gait-checkpoint), linked to `wke9245/CE-Gait`. The serialized weights, optimizer states, iteration counters, and RNG states are preserved in full.

Registry reference: `ghcr.io/wke9245/ce-gait-checkpoint:casia-b-step30000`.

The package contains `best.pt` and `best.json`. Its immutable manifest digest and file digests are recorded in `package.json`.

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

The download script reads a GitHub token with package read access from `GH_TOKEN` or `GITHUB_TOKEN`. A local token file can be selected with `--token-file`. Each file is verified before it replaces the local destination.

The checkpoint includes protection and in-loop attacker states. The frozen IHGR-Net encoder is loaded separately. Detailed metadata is stored in `best.json`.
