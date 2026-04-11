"""Central Hydra entry point for the ADMET-DKL benchmark.

Training and hyperparameter search are the same action — the difference is
whether you ask Hydra to run it once, or ask the Optuna sweeper to run it N
times.

Modes (set via mode= override):
    train  — train a single model on a single dataset, evaluate, save report.
    embed  — run offline MoLFormer embedding extraction.

Example CLI invocations:
    python main.py                                          # MLP on caco2_wang
    python main.py dataset=hia_hou model=sv_dkl
    python main.py train.lr=5e-4 model.hidden_dims=[1024,512,256]
    python main.py model.use_residual=true model.n_inducing=256
    python main.py model.hidden_dims=[]                     # 0-layer: linear/pure-GP
    python main.py mode=embed
    python main.py mode=embed dataset=all                  # all ADMET datasets

    # Optuna sweep (Hydra -m activates the optuna sweeper):
    python main.py -m dataset=caco2_wang model=sv_dkl
    python main.py -m dataset=hia_hou model=mlp hydra.sweeper.n_trials=50
    python main.py -m model=sv_dkl train.max_epochs=40     # fast sweep (40 epochs/trial)
"""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)


# hydra-optuna-sweeper currently maps to deprecated Optuna distribution classes.
# Keep sweep logs clean until the plugin updates this internally.
warnings.filterwarnings(
    "ignore",
    message=r"LogUniformDistribution has been deprecated in v3\.0\.0.*",
    category=FutureWarning,
    module=r"hydra_plugins\.hydra_optuna_sweeper\._impl",
)
warnings.filterwarnings(
    "ignore",
    message=r"UniformDistribution has been deprecated in v3\.0\.0.*",
    category=FutureWarning,
    module=r"hydra_plugins\.hydra_optuna_sweeper\._impl",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _infer_task(train_val_y: np.ndarray) -> str:
    """Infer regression vs classification from label values.

    Classification is assumed when the unique values are a subset of {0, 1}.

    Args:
        train_val_y: Raw label array from the TDC train_val split.

    Returns:
        "classification" or "regression".
    """
    unique = set(train_val_y.astype(float))
    return "classification" if unique.issubset({0.0, 1.0}) else "regression"


def _parse_hidden_dims(raw_hidden_dims: Any) -> list[int]:
    """Parse hidden-dimension config into a concrete list[int].

    Supports both direct list values (manual runs) and compact categorical
    tokens used in Optuna sweeps (e.g. "h512_256").
    """
    if isinstance(raw_hidden_dims, str):
        presets: dict[str, list[int]] = {
            "h512_256_128": [512, 256, 128],
            "h512_256": [512, 256],
            "h512": [512],
            "h256": [256],
            "h128_64": [128, 64],
            "h0": [],
        }
        if raw_hidden_dims in presets:
            return presets[raw_hidden_dims]

        # Allow string forms like "[512,256]" or "512,256" for robustness.
        text = raw_hidden_dims.strip()
        if text in {"", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        return [int(tok.strip()) for tok in text.split(",") if tok.strip()]

    return [int(v) for v in list(raw_hidden_dims)]


def _sample_inducing_init_inputs(
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    n_points: int,
    seed: int,
) -> Tensor | None:
    """Sample raw training inputs to initialise DKL inducing points.

    Pulls points directly from the underlying in-memory embedding tensor when
    available (EmbeddingDataset/Subset), avoiding a random-rank mismatch between
    latent features and inducing locations at epoch 0.
    """
    dataset = train_loader.dataset

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    if not hasattr(base_dataset, "X"):
        return None

    X_all = getattr(base_dataset, "X")
    if not isinstance(X_all, Tensor):
        return None

    idx_tensor = torch.as_tensor(indices, dtype=torch.long)
    if idx_tensor.numel() == 0:
        return None

    take = min(int(n_points), int(idx_tensor.numel()))
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(int(idx_tensor.numel()), generator=generator)[:take]
    picked = idx_tensor[perm]
    return X_all[picked].clone()


def _build_model(
    cfg: DictConfig,
    input_dim: int,
    task: str,
    train_loader: DataLoader[tuple[Tensor, Tensor]] | None = None,
) -> Any:
    """Instantiate the model specified by cfg.model.type.

    Args:
        cfg:       Full resolved Hydra config.
        input_dim: MoLFormer embedding dimension (768).
        task:      "regression" or "classification" (inferred at runtime).
        train_loader: Optional train loader used to initialise DKL inducing points.

    Returns:
        Instantiated MLPPredictor or DKLModel.
    """
    m = cfg.model
    hidden_dims = _parse_hidden_dims(m.hidden_dims)  # may be [] for 0-layer (linear/pure-GP)

    if m.type == "mlp":
        from src.models.baseline_mlp import MLPPredictor

        return MLPPredictor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_batchnorm=m.use_batchnorm,
            dropout_rate=m.dropout_rate,
            use_residual=m.use_residual,
        )
    elif m.type == "sv_dkl":
        from src.models.sv_dkl import DKLModel

        inducing_init_inputs: Tensor | None = None
        if train_loader is not None:
            inducing_init_inputs = _sample_inducing_init_inputs(
                train_loader=train_loader,
                n_points=int(m.n_inducing),
                seed=int(cfg.seed),
            )

        return DKLModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_inducing=m.n_inducing,
            use_batchnorm=m.use_batchnorm,
            dropout_rate=m.dropout_rate,
            use_residual=m.use_residual,
            task=task,
            inducing_init_inputs=inducing_init_inputs,
        )
    else:
        raise ValueError(f"Unknown model type: {m.type!r}")


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def _run_train(cfg: DictConfig) -> float:
    """Full train + evaluate + report pipeline for one model/dataset.

    Returns the validation metric (MAE for regression, 1-ROC-AUC for
    classification) so that Optuna can minimise it when running as a sweep.
    Single runs also return this value (Hydra ignores it outside -m mode).
    """
    from sklearn.metrics import roc_auc_score
    from tdc.benchmark_group import admet_group

    from src.data.tdc_datamodule import ADMETDataModule
    from src.train import train_dkl, train_mlp
    from src.utils.metrics import (
        area_under_rejection_curve,
        evaluate_tdc,
        gaussian_nll,
        picp,
        rejection_curve,
    )
    from src.utils.reporting import plot_rejection_curve, plot_training_curves, save_mini_report

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # Works for both single run (hydra.run.dir) and multirun (sweep.dir/subdir)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data — load benchmark and infer task from label distribution
    group = admet_group(path=cfg.data.tdc_group_path)
    benchmark = group.get(cfg.dataset)
    task = _infer_task(benchmark["train_val"]["Y"].values)
    log.info("Dataset: %s  task: %s", cfg.dataset, task)

    dm = ADMETDataModule(
        dataset_name=cfg.dataset,
        emb_root=cfg.data.emb_root,
        val_fraction=cfg.data.val_fraction,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    loaders = dm.setup()
    log.info(
        "Data: train=%d  val=%d  test=%d  dim=%d",
        loaders.n_train,
        loaders.n_val,
        loaders.n_test,
        loaders.input_dim,
    )

    # 2. Model
    model = _build_model(cfg, loaders.input_dim, task, train_loader=loaders.train)

    # 3. Train
    if cfg.model.type == "mlp":
        result = train_mlp(
            model,
            loaders.train,
            loaders.val,
            cfg.train,
            task=task,
            out_dir=out_dir,
            device=device,
        )
    else:
        result = train_dkl(
            model,
            loaders.train,
            loaders.val,
            cfg.train,
            n_train=loaders.n_train,
            out_dir=out_dir,
            device=device,
        )

    is_clf = task == "classification"

    # 4. Compute val metric for Optuna objective (always minimize).
    #    Regression: val MAE.  Classification: 1 - val ROC-AUC.
    #    Cannot use group.evaluate() here — it scores against the benchmark test set.
    model.eval()
    val_mu_list, val_y_list = [], []
    with torch.no_grad():
        for X, y in loaders.val:
            X = X.to(device)
            if cfg.model.type == "mlp":
                mu, _ = model.predict_uncertainty(X, T=cfg.model.mc_samples)
                val_mu_list.append(mu.squeeze(-1).cpu())
            else:
                mu, _ = model.predict(X)
                val_mu_list.append(mu.cpu())
            val_y_list.append(y)

    val_mu = torch.cat(val_mu_list)
    val_y = torch.cat(val_y_list)

    if is_clf:
        probs = torch.sigmoid(val_mu).numpy() if cfg.model.type == "mlp" else val_mu.numpy()
        val_metric = 1.0 - float(roc_auc_score(val_y.numpy(), probs))
    else:
        val_metric = float((val_y - val_mu).abs().mean().item())
    log.info("Val metric (Optuna obj, minimize): %.4f", val_metric)

    # 5. Predict on test set (full T=mc_samples for MLP at evaluation time)
    all_mu, all_var, all_y = [], [], []
    with torch.no_grad():
        for X, y in loaders.test:
            X = X.to(device)
            if cfg.model.type == "mlp":
                mu, var = model.predict_uncertainty(X, T=cfg.model.mc_samples)
            else:
                mu, var = model.predict(X)
            all_mu.append(mu.squeeze(-1).cpu())
            all_var.append(var.squeeze(-1).cpu())
            all_y.append(y)

    mu_t = torch.cat(all_mu)
    var_t = torch.cat(all_var)
    y_t = torch.cat(all_y)

    # MLP outputs raw logits; convert to probabilities for classification eval
    if is_clf and cfg.model.type == "mlp":
        mu_t = torch.sigmoid(mu_t)

    # 6. TDC evaluation (primary metric: MAE or ROC-AUC)
    tdc_metrics = evaluate_tdc(group, {cfg.dataset: mu_t.numpy()})

    # 7. Uncertainty metrics (Gaussian NLL + PICP for regression; binary NLL for clf)
    if is_clf:
        p = mu_t.clamp(1e-6, 1 - 1e-6)
        nll = -(y_t * p.log() + (1 - y_t) * (1 - p).log()).mean().item()
        cov = float("nan")
    else:
        nll = gaussian_nll(y_t, mu_t, var_t)
        cov = picp(y_t, mu_t, var_t)

    fracs, vals = rejection_curve(
        y_t,
        mu_t,
        var_t,
        metric_fn="error_rate" if is_clf else "mae",
    )
    aurc = area_under_rejection_curve(fracs, vals)

    all_metrics: dict[str, float] = {**tdc_metrics, "NLL": nll, "PICP_95": cov, "AURC": aurc}
    log.info("Metrics: %s", all_metrics)

    # 8. Plots — training curves include per-epoch NLL and PICP from val set
    plot_training_curves(
        result.train_losses,
        result.val_losses,
        result.val_metrics,  # {"NLL": [...], "PICP_95": [...]}
        out_path=out_dir / "training_curves.pdf",
        title=f"{cfg.model.type} - {cfg.dataset}",
    )
    plot_rejection_curve(
        {cfg.model.type: (fracs, vals)},
        out_path=out_dir / "rejection_curve.pdf",
        ylabel="MAE" if not is_clf else "Error Rate",
        title=cfg.dataset,
    )

    # 9. Report — includes val_curves for per-epoch calibration history
    report_path = save_mini_report(
        dataset=cfg.dataset,
        model_name=cfg.model.type,
        task=task,
        metrics=all_metrics,
        val_curves=result.val_metrics,
        model=model,
        runtime_seconds=result.runtime_seconds,
        config_snapshot=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        out_dir=out_dir,
    )
    log.info("Report saved to %s", report_path)

    return val_metric


# ---------------------------------------------------------------------------
# Embed mode
# ---------------------------------------------------------------------------


def _resolve_embed_datasets(cfg: DictConfig) -> list[str]:
    """Resolve datasets for embedding extraction.

    Supports these forms:
        - ``dataset=<name>``
        - ``dataset=all``
        - ``embed_datasets=[name1,name2,...]``
        - ``embed_datasets=[all]``

    Returns:
        Deduplicated dataset names in order.
    """
    raw_embed_datasets = cfg.get("embed_datasets")

    if raw_embed_datasets is None:
        requested = [str(cfg.dataset)]
    elif isinstance(raw_embed_datasets, str):
        requested = [raw_embed_datasets]
    else:
        requested = [str(ds) for ds in raw_embed_datasets]

    if any(ds.lower() == "all" for ds in requested):
        from tdc.benchmark_group import admet_group

        group = admet_group(path=cfg.data.tdc_group_path)
        requested = [str(name) for name in group.dataset_names]

    # Preserve order while removing duplicates.
    datasets: list[str] = []
    seen: set[str] = set()
    for ds in requested:
        if ds not in seen:
            datasets.append(ds)
            seen.add(ds)

    return datasets


def _run_embed(cfg: DictConfig) -> None:
    """Trigger offline MoLFormer embedding extraction."""
    datasets = _resolve_embed_datasets(cfg)
    log.info("Embedding datasets: %s", datasets)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.extract_embeddings",
            *datasets,
            "--out-dir",
            cfg.data.emb_root,
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Hydra entry point.

    Single run:  python main.py [overrides]
    Optuna sweep: python main.py -m [overrides]

    Returns the validation metric so Optuna can use it as the objective.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        return _run_train(cfg)
    elif cfg.mode == "embed":
        _run_embed(cfg)
        return None
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. Choose from ['train', 'embed']")


if __name__ == "__main__":
    main()
