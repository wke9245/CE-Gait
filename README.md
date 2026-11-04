# CE-Gait

**Cumulative-Exposure-Aware Revocable Gait Template Learning with Marginal Linkage Suppression**

CE-Gait learns key-conditioned gait templates using same-domain recognition, context-averaged marginal linkage suppression, and exposure-group optimization. The implementation includes IHGR-Net feature extraction, protected-template training, biometric hashing baselines, independent linkage auditors, and the E1--E7 experiment drivers.

## Manuscript

| Field | Information |
|---|---|
| Title | CE-Gait: Cumulative-Exposure-Aware Revocable Gait Template Learning with Marginal Linkage Suppression |
| Authors | Ke Wang and Hua Huo |
| Journal | Information Sciences |
| Manuscript ID | INS-D-26-12803 |
| Status | Under review |

Additional submission and citation information is provided in [MANUSCRIPT.md](MANUSCRIPT.md).

## Installation

Python 3.9 or later and PyTorch 2.1 or later are required. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For CUDA execution, install the PyTorch build appropriate for the available CUDA runtime. `--device auto` selects CUDA when available; `--device cpu` selects CPU execution.

## Checkpoint

The CASIA-B protector checkpoint is distributed through the [CE-Gait Release](https://github.com/wke9245/CE-Gait/releases/tag/CE-Gait). The release belongs to this private repository and contains `best.pt`, `best.json`, and `SHA256SUMS`. It records protector iteration 30,000, together with the in-loop adversary, optimizer states, and random-number states.

Authenticate with a GitHub token that has repository read access through `GH_TOKEN`, `GITHUB_TOKEN`, or `--token-file`. The download command checks the release tag and asset identifiers, then verifies each file's SHA-256 digest and byte count.

```bash
python scripts/download_checkpoint.py
python run.py checkpoint-info --checkpoint checkpoints/best.pt
python run.py audit --config configs/casia_b.yaml --checkpoint checkpoints/best.pt --output outputs/casia_b_audit
```

`checkpoints/release.json` records the release ID, tag, asset IDs, download URLs, and SHA-256 digests. Checkpoint binaries are downloaded locally and excluded from source control.

CASIA-B auditing and E3--E6 use this checkpoint by default. `--checkpoint` selects another model explicitly. E1, E2, and E7 train their configured variants. IHGR-Net encoder weights are loaded separately for silhouette feature extraction and template issuance.

## Data preparation

Obtain CASIA-B or Gait3D from the dataset provider and preprocess the silhouette sequences. The indexing command accepts the OpenGait directory layout:

```text
dataset_root/
  identity/
    sequence/
      view/
        silhouettes.pkl
```

Each `.pkl` or `.npy` file contains a silhouette array with shape `[frames, height, width]`. A view directory containing PNG silhouettes is also accepted. Sequences are binarized, resized to 64 x 44, and sampled to 30 temporally ordered frames. Historical eligibility uses the original count of nonempty frames and unique trajectory identifiers.

### CASIA-B

```bash
python run.py prepare --dataset casia_b --root /path/to/CASIA-B --output manifests/casia_b_sequences.json
python run.py pretrain --manifest manifests/casia_b_sequences.json --output outputs/ihgr_casia_b/ihgr.pt --device auto
python run.py extract --manifest manifests/casia_b_sequences.json --checkpoint outputs/ihgr_casia_b/ihgr.pt --output data/casia_b/manifest.json
python run.py train --config configs/casia_b.yaml --output outputs/casia_b
python run.py audit --config configs/casia_b.yaml --checkpoint outputs/casia_b/best.pt --output outputs/casia_b_audit
```

| Role | Identities |
|---|---|
| Protector and in-loop adversary | 001--030 |
| Independent auditor | 031--060 |
| Validation | 061--068 |
| Threshold calibration | 069--074 |
| Evaluation | 075--124 |

Encoder pretraining uses the 74 development identities. Recognition uses NM#1--4 as gallery sequences and NM#5--6, BG#1--2, and CL#1--2 as probes, with identical-view comparisons excluded.

### Gait3D

Supply the official `Gait3D.json` identity and probe partition. The first 2,500 sorted development identities are assigned to protector and auditor fitting, the next 300 to validation, and the remaining 200 to calibration. The official 1,000 evaluation identities are kept separate.

```bash
python run.py prepare --dataset gait3d --root /path/to/Gait3D/silhouettes --partition /path/to/Gait3D.json --output manifests/gait3d_sequences.json
python run.py pretrain --manifest manifests/gait3d_sequences.json --output outputs/ihgr_gait3d/ihgr.pt --device auto
python run.py extract --manifest manifests/gait3d_sequences.json --checkpoint outputs/ihgr_gait3d/ihgr.pt --output data/gait3d/manifest.json
python run.py train --config configs/gait3d.yaml --output outputs/gait3d
python run.py audit --config configs/gait3d.yaml --checkpoint outputs/gait3d/best.pt --output outputs/gait3d_audit
```

Recognition retains the official gallery/probe assignment. Historical linkage cohorts are selected from identities with enough distinct, usable tracklets for the specified history and a separate candidate.

## Training configuration

The dataset configurations are in `configs/casia_b.yaml` and `configs/gait3d.yaml`.

| Setting | Default |
|---|---:|
| Protector iterations | 80,000 |
| Utility warm-up iterations | 10,000 |
| Attacker updates per protector update | 2 |
| Linkage episodes per step | 32 |
| Utility identities x clips | 8 x 4 |
| Contexts per record | 4 |
| Protector / attacker learning rate | 1e-4 / 2e-4 |
| Marginal weight / group temperature | 1.0 / 0.2 |
| Privacy / quantization weight | 0.5 / 0.1 |
| Template components / bits per component | 256 / 4 |

The pretrained encoder stays frozen during protection training. The protector and adversary use AdamW with cosine learning-rate annealing. Validation identities and independent domain-key schedules determine checkpoint selection.

```bash
python run.py train --config configs/casia_b.yaml --seed 1337 --output outputs/casia_b_seed1337
python run.py train --config configs/casia_b.yaml --seed 2024 --output outputs/casia_b_seed2024
python run.py train --config configs/casia_b.yaml --resume outputs/casia_b/last.pt --output outputs/casia_b
```

During training, `best.pt` is saved when the validation objective improves. `last.pt` records the most recent saved iteration. Both files contain optimizer and random-number states for continuation.

## Experiments

| Entry point | Evaluation |
|---|---|
| `experiments/e1_capacity_ablation.py` | Capacity controls, key placement, and objective ablations |
| `experiments/e2_baselines.py` | Unprotected features and template-protection baselines |
| `experiments/e3_candidate_variation.py` | Candidate viewpoint, carrying, and clothing changes |
| `experiments/e4_long_history.py` | Extended histories and repeated renewal |
| `experiments/e5_independent_attacks.py` | Independent set auditors and attacker knowledge conditions |
| `experiments/e6_cohort_coverage.py` | Eligible identity coverage and cohort recognition |
| `experiments/e7_sensitivity.py` | Context budget, objective weights, and update-ratio sensitivity |

```bash
python experiments/e1_capacity_ablation.py --config configs/casia_b.yaml
python experiments/e2_baselines.py --config configs/gait3d.yaml
python experiments/e3_candidate_variation.py --config configs/casia_b.yaml --checkpoint outputs/casia_b/best.pt
python experiments/e4_long_history.py --config configs/casia_b.yaml --checkpoint outputs/casia_b/best.pt
python experiments/e5_independent_attacks.py --config configs/gait3d.yaml --checkpoint outputs/gait3d/best.pt
python experiments/e6_cohort_coverage.py --config configs/gait3d.yaml --checkpoint outputs/gait3d/best.pt
python experiments/e7_sensitivity.py --config configs/casia_b.yaml
```

E1, E2, and E7 train their respective variants. E3--E6 use `--checkpoint` or the checkpoint set in the dataset configuration. A protector is trained when neither source is configured. E3 uses the CASIA-B observation configurations. Each experiment writes its own configuration and result summary under `outputs/`.

## Independent security evaluation

```bash
python run.py audit --config configs/casia_b.yaml --checkpoint outputs/casia_b/best.pt --kind deepsets --output outputs/casia_b_deepsets
python run.py audit --config configs/gait3d.yaml --checkpoint outputs/gait3d/best.pt --kind transformer --knowledge disclosed --output outputs/gait3d_disclosed
python run.py audit --config configs/gait3d.yaml --checkpoint outputs/gait3d/best.pt --kind transformer --knowledge auxiliary --output outputs/gait3d_auxiliary
```

Auditors use independent initialization and validation-based selection. Available knowledge conditions are `secret`, `disclosed`, `disclosure_control`, `auxiliary`, and `auxiliary_control`. Key-disclosure conditions use DeepSets or Set Transformer auditors.

Each evaluation stores observation-level episode manifests, per-episode CSV scores, calibrated operating thresholds, per-schedule AUCs, and summary metrics. AUCs are averaged across ten domain-key schedules. Confidence intervals use 1,000 identity-cluster and key-schedule BCa resamples. Calibration identities determine the nominal 1% FPR threshold; evaluation reports both TPR and realized FPR.

Paired score comparisons use aligned episode and schedule identifiers:

```bash
python scripts/compare_scores.py outputs/method_a/m4/scores.csv outputs/method_b/m4/scores.csv --output outputs/paired_difference.json
```

## Protected-template interface

```python
from cegait.api import CEGaitPipeline
from cegait.data import load_sequence
from cegait.protocols.domain_key_schedule import new_domain_key

pipeline = CEGaitPipeline.from_checkpoints(
    "outputs/ihgr_casia_b/ihgr.pt",
    "outputs/casia_b/best.pt",
    device="cpu",
)
silhouettes, _ = load_sequence("/path/to/silhouettes.pkl")
domain_key = new_domain_key()
issued = pipeline.issue(silhouettes.unsqueeze(0), domain_key)
payload = issued["payload"]
```

One 128-bit key is shared across identities in a protection domain. The payload is a `uint8` array with shape `[batch, 128]`. Domain renewal uses a fresh key and trusted enrollment observations; gallery and probe templates are issued under the same active key. Historical records are not inputs to template issuance.

The command-line issuance interface reads a 16-byte binary domain key:

```bash
python run.py issue --backbone outputs/ihgr_casia_b/ihgr.pt --checkpoint outputs/casia_b/best.pt --sequence /path/to/silhouettes.pkl --key-file /path/to/domain.key --output outputs/template.npy
```

## Repository layout

```text
cegait/
  backbone/       IHGR-Net, SGRA, SKA, and frozen adapter
  models/         Key-conditioned protector and set auditors
  losses/         Utility, quantization, marginal, and exposure objectives
  protocols/      Observation episodes and domain-key schedules
  training/       Encoder, alternating protector, and independent auditor fitting
  baselines/      Biometric hashing and protected-template controls
  evaluation/     Recognition, linkage, cohort analysis, and BCa statistics
configs/          Dataset and auditor configurations
checkpoints/      Release reference and checkpoint metadata
experiments/      E1--E7 entry points
scripts/          Checkpoint download and paired statistical analysis
```

Data formats are specified in [DATA_FORMAT.md](DATA_FORMAT.md). References are listed in [REFERENCES.md](REFERENCES.md).
