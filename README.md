# ADMET-DKL

Benchmarks **Stochastic Variational Deep Kernel Learning (SV-DKL)** against an **MLP with Monte Carlo Dropout** on the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/). Both models encode molecules via frozen [MoLFormer-XL](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) 768-D embeddings extracted offline.

The core claim: on scaffold-split test sets (OOD regime), the GP posterior provides superior epistemic uncertainty calibration (lower NLL, higher PICP, lower AURC) compared to MC Dropout.

---

## Project structure

```
admet-dkl/
├── main.py                          # Hydra entry point (train / tune / embed)
├── configs/                         # Hydra config tree
│   ├── config.yaml
│   ├── dataset/{caco2_wang,hia_hou}.yaml
│   ├── model/{mlp,sv_dkl}.yaml
│   └── train/default.yaml
└── src/
    ├── data/
    │   ├── extract_embeddings.py    # Offline MoLFormer encoder
    │   └── tdc_datamodule.py        # Dataset + DataLoaders
    ├── models/
    │   ├── baseline_mlp.py          # TaskHead + MLPPredictor + MC Dropout
    │   └── sv_dkl.py                # DKLModel (GPyTorch ApproximateGP)
    ├── utils/
    │   ├── metrics.py               # NLL, PICP, Rejection Curve / AURC
    │   └── reporting.py             # mini_report.json + NeurIPS-style PDFs
    ├── train.py                     # Training loops + EarlyStopping
    └── tune.py                      # Optuna objectives + run_study()
```

---

## Hardware

Tested on **NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM)**, Intel Core i7, 16 GB RAM.  
All timing estimates below are for this hardware.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
git clone <repo>
cd admet-dkl

uv python pin 3.11
uv sync --group dev        # installs all deps including torch cu126
```

> **First run only:** `uv sync` downloads ~2.5 GB (PyTorch cu126 wheel).  
> Estimated time: **5–10 min** depending on bandwidth.

---

## Step 1 — Download TDC benchmark data

TDC data is downloaded automatically on first use. To pre-fetch:

```bash
uv run python -c "
from tdc.benchmark_group import admet_group
group = admet_group(path='tdc_data')
for ds in ['caco2_wang', 'hia_hou']:
    group.get(ds)
"
```

> Estimated time: **< 1 min** (1.5 MB archive).

---

## Step 2 — Extract MoLFormer embeddings (offline, once)

Encodes all SMILES to frozen 768-D `[CLS]` vectors and saves `.pt` caches.

```bash
# Single dataset
uv run python -m src.data.extract_embeddings caco2_wang

# All datasets at once
uv run python -m src.data.extract_embeddings caco2_wang hia_hou --out-dir data/embeddings
```

Cached files written to `data/embeddings/<dataset>/{train_val,test}.pt`.

> **Model download (~500 MB):** first call downloads MoLFormer weights. Estimated: **2–3 min**.  
> **Embedding inference (RTX 3050 Ti):** ~10 s total for both datasets (910 + 578 SMILES at batch=256).

---

## Step 3 — Train

All runs are launched via `main.py` with Hydra overrides.

### MLP baseline (MC Dropout)

```bash
# Regression — Caco-2 permeability
uv run python main.py dataset=caco2_wang model=mlp

# Classification — HIA oral absorption
uv run python main.py dataset=hia_hou model=mlp
```

### SV-DKL

```bash
uv run python main.py dataset=caco2_wang model=sv_dkl
uv run python main.py dataset=hia_hou    model=sv_dkl
```

### Common overrides

```bash
# Override learning rate and architecture
uv run python main.py model=sv_dkl train.lr=5e-4 model.hidden_dims=[1024,512,256]

# Residual connections, fewer inducing points
uv run python main.py model=sv_dkl model.use_residual=true model.n_inducing=256

# CPU-only run (no CUDA)
uv run python main.py device=cpu data.num_workers=0
```

> **Training time per run (RTX 3050 Ti, default config, 200 epochs max):**  
> MLP: ~2–3 min | SV-DKL: ~8–12 min (512 inducing points dominate).  
> Early stopping (patience=20) typically triggers before 200 epochs.

Single runs write to `outputs/<dataset>/<model>/runs/<timestamp>/`:
- `best_{mlp,dkl}.pt` — best checkpoint
- `training_curves.pdf`, `rejection_curve.pdf` — NeurIPS-style plots
- `mini_report.json` — metrics, param count, runtime, config snapshot
- `tensorboard/` — TensorBoard event files
- `.hydra/` — full config dump for reproducibility

```bash
tensorboard --logdir outputs/caco2_wang/mlp/runs/
```

---

## Step 4 — Hyperparameter tuning (Optuna sweep)

Training and tuning are the same command — the `-m` flag activates the
hydra-optuna-sweeper. Each trial gets its own isolated output folder.

```bash
# 30-trial sweep (default n_trials) with the shared search space
uv run python main.py -m dataset=caco2_wang model=sv_dkl

# Override number of trials or epoch budget per trial
uv run python main.py -m dataset=hia_hou model=mlp hydra.sweeper.n_trials=50 train.max_epochs=40

# Also sweep hidden_dims (CLI override; not in YAML search space by default)
uv run python main.py -m model=sv_dkl "model.hidden_dims=choice([512,256],[256],[])"
```

Sweep trials are written to `outputs/<dataset>/<model>/sweeps/<timestamp>/<trial_num>/`.
Each trial has its own `mini_report.json`, PDF plots, and TensorBoard logs.

```bash
# View all trials in TensorBoard (overlaid curves)
tensorboard --logdir outputs/caco2_wang/sv_dkl/sweeps/
```

The Optuna objective is **val MAE** (regression) or **1 − val ROC-AUC** (classification),
always minimised. The search space is defined in `configs/config.yaml` (shared) and
`configs/model/sv_dkl.yaml` (DKL-specific: `n_inducing`, `gp_lr`).

---

## Model parameters

| Model | Parameters |
|---|---|
| MLP (512→256) | 526,849 |
| SV-DKL (512→256, M=512) | 920,580 |

---

## Evaluated metrics

| Metric | Task | Source |
|---|---|---|
| MAE | Regression | `group.evaluate()` (TDC) |
| ROC-AUC | Classification | `group.evaluate()` (TDC) |
| NLL | Regression (Gaussian) / Classification (binary) | `src/utils/metrics.py` |
| PICP @ 95% | Regression only | `src/utils/metrics.py` |
| AURC | Both | `src/utils/metrics.py` |

---

## Run tests

```bash
uv run pytest tests/ -v
```

42 tests covering dataset loading, MC Dropout correctness (BN frozen, only Dropout active), DKL forward/backward pass, and all metric functions.
