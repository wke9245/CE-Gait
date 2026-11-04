# CE-Gait

**Cumulative-Exposure-Aware Revocable Gait Template Learning with Marginal Linkage Suppression**

CE-Gait is a research implementation accompanying our manuscript submitted to *Information Sciences*. The repository presents the software architecture and principal components of the proposed cumulative-exposure-aware gait template protection framework.

The project includes the IHGR-Net feature extraction interface, key-conditioned template protection, marginal linkage suppression, exposure-group optimization, independent linkage auditing, and experimental procedures for CASIA-B and Gait3D.

The implementation is provided for academic inspection during manuscript review. The original research software and associated materials remain subject to the copyright and access conditions specified in this repository.

## Manuscript

**Journal:** Information Sciences. **Manuscript ID:** INS-D-26-12803. **Status:** Under review.

Submission details and citation information are maintained in [MANUSCRIPT.md](MANUSCRIPT.md) and [CITATION.cff](CITATION.cff).

## Software environment

- Python 3.15
- PyTorch 2.15
- CUDA 13.2 for GPU execution
- NumPy and SciPy
- PyYAML, cryptography, and Pillow

The dependency files specify the selected software environment. For GPU execution with CUDA 13.2, install PyTorch from its dedicated wheel index:

```bash
python -m pip install "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132
```

Install the remaining dependencies and the project with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Other hardware configurations can use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

A Conda environment is specified in `environment.yml`.

## Main components

| Module | Role |
|---|---|
| `cegait/backbone/` | IHGR-Net interface, spatial reliability, and temporal aggregation |
| `cegait/models/` | Key-conditioned protection and linkage auditors |
| `cegait/losses/` | Recognition, quantization, marginal linkage, and exposure objectives |
| `cegait/protocols/` | Historical observations and domain-key schedules |
| `cegait/training/` | Encoder, protector, and independent-auditor optimization |
| `cegait/baselines/` | Biometric hashing and protection controls |
| `cegait/evaluation/` | Recognition, linkage, cohort analysis, and statistical evaluation |

## Experiments

| Entry point | Experiment |
|---|---|
| `experiments/e1_capacity_ablation.py` | Capacity controls and component ablations |
| `experiments/e2_baselines.py` | Template-protection baselines |
| `experiments/e3_candidate_variation.py` | Candidate observation variation |
| `experiments/e4_long_history.py` | Extended histories and repeated renewal |
| `experiments/e5_independent_attacks.py` | Independent auditors and attacker knowledge |
| `experiments/e6_cohort_coverage.py` | Eligible identity coverage and cohort recognition |
| `experiments/e7_sensitivity.py` | Objective and optimization sensitivity |

Dataset settings are provided in `configs/casia_b.yaml` and `configs/gait3d.yaml`. Command-line options are available through:

```bash
python run.py --help
python run.py experiment --help
```

Data conventions are described in [DATA_FORMAT.md](DATA_FORMAT.md).

## Checkpoints

Model checkpoints and supporting metadata are maintained separately from the source code in [CE-Gait Releases](https://github.com/wke9245/CE-Gait/releases/tag/CE-Gait). Access to the associated research assets is managed by the authors during manuscript review. Authorized users can retrieve the assets with `scripts/download_checkpoint.py`.

## References and rights

Related software and literature are listed in [REFERENCES.md](REFERENCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The software remains subject to the [All rights reserved](LICENSE) notice.
