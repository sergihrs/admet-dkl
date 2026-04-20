# Probabilistic Modelling for Drug Discovery
Benchmarking SV-DKL against Deep Ensembles in High-Dimensional Low-Sample Regimes.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Optuna](https://img.shields.io/badge/Optuna-2.0%2B-purple.svg)
![Hydra](https://img.shields.io/badge/Hydra-1.3%2B-blueviolet.svg)

## Overview

In ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction, we often encounter the **"Curse of Dimensionality"** ($P \gg N$). High-dimensional embeddings combined with extremely small sample sizes present a significant challenge. Traditional tree-based ensembles like XGBoost suffer from a **"Mismatch in Inductive Biases"** when operating on dense, continuous latent representations. 

Stochastic Variational Deep Kernel Learning (SV-DKL) provides to combine the feature extraction capabilities of deep neural networks with the rigorous uncertainty quantification of Gaussian Processes. However, in limited data regimes, standard SV-DKL is prone to **"Feature Collapse"**—where the neural network overfits and maps all inputs to a degenerate manifold. We demonstrate that applying **Spectral Normalization** to the feature extractor resolves this collapse, preserving distance awareness and yielding superior Predictive Uncertainty calibration.

[Read the full paper here](report/paper.pdf)

## Key Results

SV-DKL with Spectral Normalization significantly outperforms XGBoost Deep Ensembles in both point accuracy and Out-of-Distribution (OOD) uncertainty calibration on the TDC `caco2_wang` dataset using MoLFormer embeddings.

| Model Architecture | MAE ↓ | NLL ↓ | PICP_95 ↑ | AURC ↓ |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Deep Ensemble** | 0.373 | 39.627 | 20.87% | 0.357 |
| **SV-DKL (No Spectral Norm)** | 0.404 | 0.753 | 91.53% | 0.369 |
| **SV-DKL (With Spectral Norm)** | **0.358** | **0.652** | **98.35%** | **0.299** |


![Rejection Curve Comparison](report/figures/spectral-norm-rc-comparison.pdf)
**Figure 1:** The Rejection Curve comparison shows that Spectral Normalization "straightens" the rejection curve, indicating that the model properly ranks predictive uncertainty and assigns higher variance to incorrect predictions.

![XGBoost Overfitting](report/figures/ensemble-training-curves.pdf)
**Figure 2:** XGBoost quickly memorizes the training set but generalizes poorly, highlighting the structural limitations of tree models on dense high-dimensional graphs compared to regularized SV-DKL.

![DKL vs XGBoost Rejection Curves](report/figures/dkl-vs-xg-rc.pdf)
**Figure 3:** SV-DKL with Spectral Normalization achieves a much steeper rejection curve compared to XGBoost, demonstrating superior uncertainty ranking and OOD detection capabilities.

## Repository Structure

```text
admet-dkl/
│
├── configs/               # Hydra configuration files
│   ├── data/              # Datamodule settings
│   ├── model/             # Model architectures (SV-DKL, XGBoost, etc.)
│   └── train/             # Training loop, optimizer, and scheduler settings
│
├── data/                  # Root folder for embeddings and raw TDC data
│   ├── embeddings/        # Pre-extracted MoLFormer embeddings (train/val/test splits)
│   └── tdc/               # Downloaded Therapeutics Data Commons datasets
│
├── outputs/               # Hydra output directory (checkpoints, sweep logs, submissions)
│
├── report/                # LaTeX source for the final research paper
│   ├── figures/           # Diagrams and plots included in the paper
│   └── sections/          # Paper sections (Introduction, Methodology, etc.)
│
├── src/                   # Main source code
│   ├── data/              # TDC datamodules and embedding extraction scripts
│   ├── models/            # PyTorch models (sv_dkl.py, baseline_mlp.py)
│   └── utils/             # Metrics, evaluation, and RC generation tools
│
├── tests/                 # Unit tests for datamodules, models, and metrics
│
├── main.py                # Main experiment execution script
├── train.py               # PyTorch training routine for neural models
├── train_xgb.py           # Training routine for XGBoost models and ensembles
└── pyproject.toml         # Python environment configuration and dependencies
```

## Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/sergihrs/admet-dkl.git
cd admet-dkl

# Create venv and install dependencies
uv sync
```

## Reproducing the Experiments

All experiments are managed using [Hydra](https://hydra.cc/) and tracked/optimized with [Optuna](https://optuna.org/).

**1. Run a single SV-DKL model:**
```bash
python main.py dataset=caco2_wang model=sv_dkl
```

**2. Run the massive Optuna sweep for SV-DKL:**
```bash
# Sweeps across structural parameters (e.g., hidden layers, spectral norm, dropout)
python main.py -m dataset=caco2_wang model=sv_dkl hydra.sweeper.n_trials=200 model.use_spectral_norm=true
```

**3. Run the XGBoost baseline sweep:**
```bash
python main.py -m dataset=caco2_wang model=xgb hydra.sweeper.n_trials=200
```