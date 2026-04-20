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

import json
import logging
import subprocess
import sys
import time
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
    print(
        f"Building model: type={m.type}  hidden_dims={hidden_dims}  n_inducing={getattr(m, 'n_inducing', 'N/A')}"
    )

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
            use_spectral_norm=bool(m.get("use_spectral_norm", False)),
            kernel_type=str(m.get("kernel_type", "rbf")),
            matern_nu=float(m.get("matern_nu", 2.5)),
            task=task,
            inducing_init_inputs=inducing_init_inputs,
        )
    else:
        raise ValueError(f"Unknown model type: {m.type!r}")


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate_split(
    model: Any,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    model_type: str,
    mc_samples: int,
    is_clf: bool,
    device: torch.device,
    dataset: str,
    group: object | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Predict on a split and compute metrics + rejection-curve payload.

    The primary metric is always the dataset's official TDC metric (from
    ``bm_metric_names``). For the test split (``group`` provided) it comes
    from ``group.evaluate()`` to match the canonical rounded benchmark
    number; for other splits it is computed with TDC's ``Evaluator``
    directly — ``group.evaluate(testing=False)`` is broken upstream.
    """
    from src.utils.metrics import (
        area_under_rejection_curve,
        evaluate_tdc,
        gaussian_nll,
        picp,
        primary_metric_for_dataset,
        rejection_curve,
        tdc_evaluator_value,
    )

    model.eval()
    mu_list, var_list, y_list = [], [], []
    for X, y in loader:
        X = X.to(device)
        if model_type == "mlp":
            mu, var = model.predict_uncertainty(X, T=mc_samples)
        else:
            mu, var = model.predict(X)
        mu_list.append(mu.squeeze(-1).cpu())
        var_list.append(var.squeeze(-1).cpu())
        y_list.append(y)

    mu_t = torch.cat(mu_list)
    var_t = torch.cat(var_list)
    y_t = torch.cat(y_list)

    # MLP outputs raw logits for classification; convert to probabilities
    if is_clf and model_type == "mlp":
        mu_t = torch.sigmoid(mu_t)

    metric_name = primary_metric_for_dataset(dataset)
    if group is not None:
        primary = evaluate_tdc(group, {dataset: mu_t.numpy()})
    else:
        val = tdc_evaluator_value(y_t.numpy(), mu_t.numpy(), metric_name)
        primary = {metric_name.upper(): val}

    if is_clf:
        p = mu_t.clamp(1e-6, 1 - 1e-6)
        nll = -(y_t * p.log() + (1 - y_t) * (1 - p).log()).mean().item()
        cov = float("nan")
    else:
        nll = gaussian_nll(y_t, mu_t, var_t)
        cov = picp(y_t, mu_t, var_t)

    reject_metric = "error_rate" if is_clf else "mae"
    fracs, vals = rejection_curve(y_t, mu_t, var_t, metric_fn=reject_metric)
    aurc = area_under_rejection_curve(fracs, vals)

    metrics = {**primary, "NLL": nll, "PICP_95": cov, "AURC": aurc}
    rej = {
        "metric": reject_metric,
        "x": [float(v) for v in fracs],
        "y": [float(v) for v in vals],
    }
    return metrics, rej


def _run_train(cfg: DictConfig) -> float:
    """Full train + evaluate + report pipeline for one model/dataset.

    Returns the validation metric (MAE for regression, 1-ROC-AUC for
    classification) so that Optuna can minimise it when running as a sweep.
    Single runs also return this value (Hydra ignores it outside -m mode).
    """
    from tdc.benchmark_group import admet_group

    from src.data.tdc_datamodule import ADMETDataModule
    from src.train import train_dkl, train_mlp
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

    from src.utils.metrics import higher_is_better, primary_metric_for_dataset

    metric_name = primary_metric_for_dataset(cfg.dataset)
    primary_key = metric_name.upper()
    eval_mc_samples = int(OmegaConf.select(cfg, "model.mc_samples", default=30))

    # 4. Evaluate on val (TDC Evaluator — group.evaluate(testing=False) is broken)
    val_metrics, val_rej = _evaluate_split(
        model,
        loaders.val,
        model_type=cfg.model.type,
        mc_samples=eval_mc_samples,
        is_clf=is_clf,
        device=device,
        dataset=cfg.dataset,
    )
    primary_val = float(val_metrics[primary_key])
    val_objective = (1.0 - primary_val) if higher_is_better(metric_name) else primary_val
    log.info("Val metrics: %s", val_metrics)
    log.info("Val objective (Optuna minimize): %.4f", val_objective)

    # 5. Evaluate on test (TDC primary metric via group.evaluate — rounded)
    test_metrics, test_rej = _evaluate_split(
        model,
        loaders.test,
        model_type=cfg.model.type,
        mc_samples=eval_mc_samples,
        is_clf=is_clf,
        device=device,
        dataset=cfg.dataset,
        group=group,
    )
    log.info("Test metrics: %s", test_metrics)

    # 6. Plots — training curves include per-epoch NLL and PICP from val set
    plot_training_curves(
        result.train_losses,
        result.val_losses,
        result.val_metrics,  # {"NLL": [...], "PICP_95": [...]}
        out_path=out_dir / "training_curves.pdf",
        title=f"{cfg.model.type} - {cfg.dataset}",
    )
    plot_rejection_curve(
        {cfg.model.type: (test_rej["x"], test_rej["y"])},
        out_path=out_dir / "rejection_curve.pdf",
        ylabel="MAE" if not is_clf else "Error Rate",
        title=cfg.dataset,
        metric_name=test_rej.get("metric"),
    )

    # 7. Report — val_metrics + test_metrics; val_curves keep per-epoch history
    report_path = save_mini_report(
        dataset=cfg.dataset,
        model_name=cfg.model.type,
        task=task,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_curves=result.val_metrics,
        model=model,
        runtime_seconds=result.runtime_seconds,
        config_snapshot=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        rejection_curve_test=test_rej,
        rejection_curve_val=val_rej,
        out_dir=out_dir,
    )
    log.info("Report saved to %s", report_path)

    return val_objective


# ---------------------------------------------------------------------------
# XGBoost modes — stage 1 (single trial for Optuna) and stage 2 (ensemble)
# ---------------------------------------------------------------------------


def _load_xgb_splits(
    cfg: DictConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    object,
]:
    """Load benchmark + in-memory arrays + infer task. Shared by stage 1/2."""
    from tdc.benchmark_group import admet_group

    from src.data.tdc_datamodule import ADMETDataModule
    from src.train_xgb import extract_arrays

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
    X_tr, y_tr = extract_arrays(loaders.train)
    X_val, y_val = extract_arrays(loaders.val)
    X_test, y_test = extract_arrays(loaders.test)
    log.info(
        "Arrays: train=%d  val=%d  test=%d  dim=%d",
        len(X_tr),
        len(X_val),
        len(X_test),
        X_tr.shape[1],
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test, task, group


def _run_xgb_single(cfg: DictConfig) -> float:
    """Stage 1: fit one XGBoost model and return the validation metric.

    Used by Optuna as the sweep objective. Minimises val MAE (regression) or
    1 - val ROC-AUC (classification).
    """
    from src.train_xgb import train_single_xgb
    from src.utils.metrics import (
        higher_is_better,
        primary_metric_for_dataset,
        tdc_evaluator_value,
    )

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_val, y_val, X_test, _y_test, task, _group = _load_xgb_splits(cfg)

    res = train_single_xgb(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_test,
        cfg.model,
        task,
        int(cfg.seed),
    )

    metric_name = primary_metric_for_dataset(cfg.dataset)
    primary_val = tdc_evaluator_value(np.asarray(y_val), np.asarray(res.val_mu), metric_name)
    val_metric = (1.0 - primary_val) if higher_is_better(metric_name) else primary_val
    log.info(
        "XGB val %s=%.4f  Optuna obj=%.4f",
        metric_name.upper(),
        primary_val,
        val_metric,
    )

    # Minimal per-trial record so sweep outputs stay introspectable.
    (out_dir / "trial_summary.json").write_text(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "task": task,
                "val_metric": val_metric,
                "n_trees": res.n_trees,
                "runtime_seconds": round(res.runtime_seconds, 2),
                "params": OmegaConf.to_container(cfg.model, resolve=True),
            },
            indent=2,
        )
    )

    return val_metric


def _run_xgb_ensemble(cfg: DictConfig) -> None:
    """Stage 2: train N XGBoost members and aggregate into an ensemble.

    Diversity comes from per-member seeds (``cantor_pairing(cfg.seed, i)``)
    combined with the tuned ``subsample`` / ``colsample_bytree`` knobs. The
    ensemble mean feeds the deterministic TDC metric; the ensemble variance
    is passed to ``gaussian_nll`` / ``picp`` / AURC as the epistemic
    uncertainty.
    """
    from src.train_xgb import cantor_pairing, train_single_xgb
    from src.utils.metrics import (
        area_under_rejection_curve,
        evaluate_tdc,
        gaussian_nll,
        picp,
        primary_metric_for_dataset,
        rejection_curve,
        tdc_evaluator_value,
    )
    from src.utils.reporting import plot_ensemble_training_curves, plot_rejection_curve

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_val, y_val, X_test, y_test, task, group = _load_xgb_splits(cfg)
    is_clf = task == "classification"

    n_members = int(cfg.model.n_members)
    log.info("Training %d XGBoost ensemble members", n_members)

    test_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    member_test_mae: list[float] = []
    train_curves: list[list[float]] = []
    val_curves: list[list[float]] = []
    total_trees = 0
    t0 = time.perf_counter()

    for i in range(n_members):
        seed_i = cantor_pairing(int(cfg.seed), i)
        log.info("[%d/%d] seed=%d", i + 1, n_members, seed_i)
        res = train_single_xgb(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            cfg.model,
            task,
            seed_i,
        )
        test_preds.append(res.test_mu)
        val_preds.append(res.val_mu)
        if not is_clf:
            member_test_mae.append(float(np.abs(y_test - res.test_mu).mean()))
        train_curves.append(res.train_curve)
        val_curves.append(res.val_curve)
        total_trees += res.n_trees

    runtime = time.perf_counter() - t0

    metric_name = primary_metric_for_dataset(cfg.dataset)

    def _ensemble_metrics(
        preds_stack: np.ndarray,
        y_split: np.ndarray,
        use_tdc: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Aggregate per-member preds into mean/var and compute the metric block."""
        mu = preds_stack.mean(axis=0)
        var = preds_stack.var(axis=0, ddof=0).clip(min=1e-8)
        y_t = torch.from_numpy(y_split).float()
        mu_t = torch.from_numpy(mu).float()
        var_t = torch.from_numpy(var).float()

        if use_tdc:
            primary = evaluate_tdc(group, {cfg.dataset: mu})
        else:
            primary = {metric_name.upper(): tdc_evaluator_value(y_split, mu, metric_name)}

        if is_clf:
            p = mu_t.clamp(1e-6, 1 - 1e-6)
            nll = -(y_t * p.log() + (1 - y_t) * (1 - p).log()).mean().item()
            cov = float("nan")
        else:
            nll = gaussian_nll(y_t, mu_t, var_t)
            cov = picp(y_t, mu_t, var_t)
        reject_metric = "error_rate" if is_clf else "mae"
        fracs, vals = rejection_curve(y_t, mu_t, var_t, metric_fn=reject_metric)
        aurc = area_under_rejection_curve(fracs, vals)
        return (
            {**primary, "NLL": nll, "PICP_95": cov, "AURC": aurc},
            {
                "metric": reject_metric,
                "x": [float(v) for v in fracs],
                "y": [float(v) for v in vals],
            },
        )

    val_metrics, val_rej = _ensemble_metrics(np.stack(val_preds, axis=0), y_val, use_tdc=False)
    test_metrics, test_rej = _ensemble_metrics(np.stack(test_preds, axis=0), y_test, use_tdc=True)
    log.info("Val metrics:  %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)

    plot_ensemble_training_curves(
        train_curves,
        val_curves,
        out_path=out_dir / "ensemble_training_curves.pdf",
        title=f"xgboost ensemble - {cfg.dataset}",
        ylabel="MAE" if not is_clf else "logloss",
    )
    plot_rejection_curve(
        {"xgb_ensemble": (test_rej["x"], test_rej["y"])},
        out_path=out_dir / "rejection_curve.pdf",
        ylabel="MAE" if not is_clf else "Error Rate",
        title=cfg.dataset,
        metric_name=test_rej.get("metric"),
    )

    report = {
        "dataset": cfg.dataset,
        "model": "xgboost_ensemble",
        "task": task,
        "n_members": n_members,
        "member_test_mae": member_test_mae,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "rejection_curve_test": test_rej,
        "rejection_curve_val": val_rej,
        "runtime_seconds": round(runtime, 2),
        "total_trees": int(total_trees),
        "config_snapshot": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = out_dir / "mini_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Report saved to %s", report_path)


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

    if cfg.mode == "train":
        if cfg.model.type == "xgboost":
            return _run_xgb_single(cfg)
        return _run_train(cfg)
    elif cfg.mode == "ensemble":
        if cfg.model.type != "xgboost":
            raise ValueError(
                f"mode=ensemble is only supported for model=xgboost, got {cfg.model.type!r}"
            )
        _run_xgb_ensemble(cfg)
        return None
    elif cfg.mode == "embed":
        _run_embed(cfg)
        return None
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. Choose from ['train', 'ensemble', 'embed']")


if __name__ == "__main__":
    main()
