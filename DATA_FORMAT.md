# Data formats

## Silhouette manifest

`prepare` writes a version-1 JSON manifest with the following fields:

| Field | Description |
|---|---|
| `schema_version` | Integer `1` |
| `kind` | `silhouettes` |
| `dataset` | `casia_b` or `gait3d` |
| `identity_roles` | Mapping from identity identifiers to experimental roles |
| `records` | Observation records |

An observation record contains `id`, `identity`, `role`, `sequence`, `view`, `path`, `tracklet`, and `gallery`. The `id` and `tracklet` identify the source trajectory. `gallery` records the official Gait3D gallery assignment. Paths may be absolute or relative to the manifest directory.

Roles are `generator`, `auditor`, `validation`, `calibration`, and `test`. The last role denotes the held-out evaluation partition. An identity belongs to exactly one role for protection and security evaluation.

## Feature cache

`extract` creates a JSON feature manifest and an NPZ archive. The JSON manifest retains the observation records and identity-role mapping, sets `kind` to `features`, and records the archive path in `features`. It also records the encoder checkpoint used for extraction.

The NPZ archive contains:

| Array | Shape | Type |
|---|---|---|
| `local` | `[observations, 16, 256]` | float32 |
| `pooled` | `[observations, 256]` | float32 |

Each observation record additionally stores `frames`, the number of original nonempty silhouette frames before temporal sampling. Local descriptor order is spatial-part major, followed by temporal-segment index. All descriptor values must be finite.

## Episode manifests

An episode manifest records source observation indices, candidate indices, binary linkage labels, historical account identifiers, and domain-tuple draw indices. Positive and matched negative episodes share history composition and domain-key assignments. Distinct methods reuse the same observation protocol and random seed.

## Evaluation outputs

| File | Content |
|---|---|
| `config.json` | Effective run configuration |
| `training.jsonl` | Protector optimization and validation metrics |
| `training_episodes.json` | Independent-auditor fitting episodes and supervision budget |
| `auditor.pt` | Validation-selected independent auditor |
| `mN/episodes.json` | Evaluation episodes at history length N |
| `mN/scores.csv` | Schedule, account, episode, draw, label, and score |
| `mN/metrics.json` | Per-schedule metrics, operating thresholds, and BCa interval |
| `summary.json` | Aggregated experiment results |

## Checkpoints and payloads

Native checkpoints use `schema_version: 1`. The `kind` field is `ihgr` for encoder checkpoints and `protector` for protected-template models. Training commands save the states needed by their corresponding continuation commands.

The private GitHub Packages artifact `ce-gait-checkpoint:casia-b-step30000` contains the CASIA-B protector at saved iteration 30,000. `scripts/download_checkpoint.py` retrieves `best.pt` and `best.json` to `checkpoints/`, using the immutable manifest digest recorded in `checkpoints/package.json`. The metadata records the checkpoint byte count and SHA-256 digest. Root `SHA256SUMS` covers version-controlled source and metadata files; checkpoint binary integrity is checked against its package digest.

The `checkpoint-info` command checks strict model-state compatibility before reporting metadata. Native checkpoints are loaded with `weights_only=True`.

CE-Gait payloads contain 256 four-bit components packed in component order, with the first component in the high nibble of the first byte. Each payload occupies 128 bytes before metadata. Matching applies normalized cosine distance to dequantized components. The feature-level hashing baselines retain their configured Hamming or index-collision matching rules.
