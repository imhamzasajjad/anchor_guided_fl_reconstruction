# Anchor-Guided Reconstruction in Federated Learning

Official implementation of the paper:

**"The Privacy Peak: Non-Monotonic Leakage in Federated Learning"**

## Overview

Federated Learning (FL) enables collaborative model training without directly sharing client data. However, trained global models may still retain information about the underlying training distribution.

This repository provides the implementation of an anchor-guided reconstruction framework for studying post-hoc privacy leakage in federated learning under varying levels of client heterogeneity. The proposed method reconstructs class-representative inputs using only the final trained global model, auxiliary feature priors, and optimization-based inversion.

The framework combines:

- Partial-anchor initialization
- Auxiliary feature-bank guidance
- Multi-objective reconstruction optimization
- Multi-anchor consistency analysis

and evaluates reconstruction behavior across multiple federated heterogeneity regimes.

---

## Repository Structure

```text
.
├── federated_training/
│   └──FL_CIFAR10.py
│
├── reconstruction/
│   ├── recon_multianchor.py
│   └── ablation_study.py
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
```

---

## Experimental Setup

### Dataset

- CIFAR-10

### Federated Learning

- Algorithm: Federated Averaging (FedAvg)
- Model: ResNet-34
- Number of Classes: 10

### Heterogeneity Settings

| Experiment | Description | Dirichlet α |
|------------|-------------|-------------|
| E1 | IID | IID |
| E2 | Mild Non-IID | 1.0 |
| E3 | Strong Non-IID | 0.5 |
| E4 | Extreme Non-IID | 0.1 |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/imhamzasajjad/anchor_guided_fl_reconstruction.git
cd anchor_guided_fl_reconstruction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Federated Models

Train a federated model under a selected heterogeneity setting:

```bash
python FL_CIFAR10.py 
```

Available experiments:

```text
E1  IID
E2  Mild Non-IID
E3  Strong Non-IID
E4  Extreme Non-IID
```

---

## Reconstruction

Run anchor-guided reconstruction using a trained global model:

```bash
python recon_multianchor.py
You have to set the checkpoint path. 
```

Outputs include:

- Reconstructed images
- Nearest-neighbour comparisons
- Per-image reconstruction metrics
- Anchor-consistency analysis
- Summary statistics

---

## Threat Model

We consider a post-hoc server-side adversary with access only to the final trained global model.

The adversary:

- Has access to model parameters
- Has access to Batch Normalization statistics
- Can query model outputs
- Does not access client datasets
- Does not access local gradients or updates
- Does not use training-time information

The attack reconstructs class-representative inputs directly from the trained global model.

---

## Evaluation Metrics

Reconstruction quality is evaluated using:

- Classification Confidence
- Prediction Entropy
- Structural Similarity Index (SSIM)
- Feature-Space Distance
- Nearest-Neighbour Similarity
- Anchor Consistency Metrics

---

## Reproducibility

All experiments use:

- Fixed random seeds
- Identical model architectures
- Identical optimization settings

Differences between experiments arise solely from data partitioning controlled by the Dirichlet parameter α.

---

## Citation

If you find this work useful, please cite:
```

---

## License

This project is released under the MIT License.
