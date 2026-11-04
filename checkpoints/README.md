# Checkpoint

`best.pt` is the supplied CASIA-B CE-Gait checkpoint at protector iteration 30,000. The serialized weights, optimizer states, iteration counters, and RNG states are preserved in full.

| Component | Recorded value |
|---|---:|
| Protector parameters | 2,368,257 |
| In-loop adversary parameters | 788,225 |
| Protector optimizer updates | 30,000 |
| Adversary optimizer updates | 40,000 |
| File size | 37,940,917 bytes |

SHA-256: `6fec75c388e3343e276faded78fbb8a26b743c10c3589cf10f01aaebcbe54933`

```bash
python run.py checkpoint-info --checkpoint checkpoints/best.pt
```

The checkpoint includes protection and in-loop attacker states. The frozen IHGR-Net encoder is loaded separately. Detailed metadata is stored in `best.json`.
