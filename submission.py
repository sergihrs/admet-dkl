"""Run official-style ADMET submissions with fixed SV-DKL hyperparameters.

This script intentionally avoids Hydra and builds a ``DictConfig`` directly
from the default YAML files under ``configs/`` plus fixed overrides.

Usage examples:
    uv run python submission.py all
    uv run python submission.py caco2_wang hia_hou
    uv run python submission.py caco2_wang --repetitions 5
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.data.tdc_datamodule import ADMETDataModule
from src.models.sv_dkl import DKLModel
from src.train import TrainResult, train_dkl

HIDDEN_DIM_PRESETS: dict[str, list[int]] = {
    "h512_256_128": [512, 256, 128],
    "h512_256": [512, 256],
    "h512": [512],
    "h256": [256],
    "h128_64": [128, 64],
    "h0": [],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace with selected datasets, repetitions, and output directory.
    """
    parser = argparse.ArgumentParser(description="Run SV-DKL submission loops.")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset names or a single 'all'.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of seed repetitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "submissions",
        help="Root directory for submission artifacts.",
    )
    return parser.parse_args()


def build_submission_cfg() -> DictConfig:
    """Create submission config from defaults + fixed overrides.

    Returns:
        DictConfig containing ``model``, ``train``, ``data``, ``seed``,
        and ``device`` sections.
    """
    root_cfg = OmegaConf.load("configs/config.yaml")
    model_cfg = OmegaConf.load("configs/model/sv_dkl.yaml")
    train_cfg = OmegaConf.load("configs/train/default.yaml")
    data_cfg = OmegaConf.load("configs/data/default.yaml")

    base_cfg = OmegaConf.create(
        {
            "seed": root_cfg.seed,
            "device": root_cfg.device,
            "model": model_cfg.model,
            "train": train_cfg,
            "data": data_cfg,
        }
    )

    overrides = OmegaConf.create(
        {
            "model": {
                "n_inducing": 256,
                "use_batchnorm": True,
                "dropout_rate": 0.19535639837860636,
                "use_residual": False,
                "hidden_dims": "h512_256",
            },
            "train": {
                "gp_lr": 0.06077752885531628,
                "lr": 0.0017661476381697725,
            },
        }
    )
    return OmegaConf.merge(base_cfg, overrides)


def parse_hidden_dims(raw_hidden_dims: Any) -> list[int]:
    """Parse hidden-dimension config into list[int]."""
    if isinstance(raw_hidden_dims, str):
        if raw_hidden_dims in HIDDEN_DIM_PRESETS:
            return HIDDEN_DIM_PRESETS[raw_hidden_dims]

        text = raw_hidden_dims.strip()
        if text in {"", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        return [int(tok.strip()) for tok in text.split(",") if tok.strip()]

    return [int(v) for v in list(raw_hidden_dims)]


def infer_task(train_val_y: np.ndarray) -> str:
    """Infer task type from labels.

    Args:
        train_val_y: Label vector from benchmark train_val split.

    Returns:
        ``classification`` when values are in {0,1}, else ``regression``.
    """
    unique = set(train_val_y.astype(float))
    return "classification" if unique.issubset({0.0, 1.0}) else "regression"


def sample_inducing_init_inputs(
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    n_points: int,
    seed: int,
) -> Tensor | None:
    """Sample raw training inputs to initialise DKL inducing points.

    Args:
        train_loader: Training loader from ADMETDataModule.
        n_points: Number of points to sample.
        seed: Sampling seed.

    Returns:
        Tensor of shape [M, input_dim] or None if unavailable.
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


def resolve_datasets(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    """Resolve user-provided dataset selection.

    Args:
        requested: CLI positional values.
        available: Benchmark dataset names from TDC.

    Returns:
        Concrete ordered list of dataset names to run.

    Raises:
        ValueError: If unknown datasets are requested.
    """
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(available)

    available_set = set(available)
    unknown = [name for name in requested if name not in available_set]
    if unknown:
        raise ValueError(
            f"Unknown datasets: {unknown}. Available datasets: {sorted(available_set)}"
        )

    # Keep user-provided order but deduplicate.
    resolved: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name not in seen:
            resolved.append(name)
            seen.add(name)
    return resolved


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible train/val splits and model init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_dataset(
    model: DKLModel,
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> np.ndarray:
    """Predict on the test set and return a NumPy array."""
    model.eval()
    preds: list[Tensor] = []

    for X, _ in test_loader:
        mu, _ = model.predict(X.to(device))
        preds.append(mu.cpu())

    return torch.cat(preds).numpy()


def run_one_training(
    cfg: DictConfig,
    dataset_name: str,
    seed: int,
    output_dir: Path,
    group: Any,
) -> tuple[np.ndarray, TrainResult, str]:
    """Train a single seed for a dataset and return predictions.

    Args:
        cfg: Submission DictConfig.
        dataset_name: TDC dataset name.
        seed: Random seed for split/model init.
        output_dir: Per-run artifact directory.
        group: tdc admet_group object.

    Returns:
        Tuple with test predictions, training result, and task name.
    """
    set_global_seed(seed)

    benchmark = group.get(dataset_name)
    task = infer_task(benchmark["train_val"]["Y"].values)

    dm = ADMETDataModule(
        dataset_name=dataset_name,
        emb_root=cfg.data.emb_root,
        val_fraction=cfg.data.val_fraction,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=seed,
    )
    loaders = dm.setup()

    hidden_dims = parse_hidden_dims(cfg.model.hidden_dims)
    inducing_init_inputs = sample_inducing_init_inputs(
        train_loader=loaders.train,
        n_points=int(cfg.model.n_inducing),
        seed=seed,
    )

    model = DKLModel(
        input_dim=loaders.input_dim,
        hidden_dims=hidden_dims,
        n_inducing=int(cfg.model.n_inducing),
        use_batchnorm=bool(cfg.model.use_batchnorm),
        dropout_rate=float(cfg.model.dropout_rate),
        use_residual=bool(cfg.model.use_residual),
        task=task,
        inducing_init_inputs=inducing_init_inputs,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_dkl(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        cfg=cfg.train,
        n_train=loaders.n_train,
        out_dir=output_dir,
        device=device,
    )

    predictions = predict_dataset(model, loaders.test, device)
    return predictions, result, task


def save_run_artifacts(
    run_dir: Path,
    predictions: np.ndarray,
    train_result: TrainResult,
    dataset_name: str,
    task: str,
) -> None:
    """Persist per-seed artifacts in the requested output folder."""
    np.save(run_dir / "test_predictions.npy", predictions)

    summary = {
        "dataset": dataset_name,
        "task": task,
        "best_val_loss": float(train_result.best_val_loss),
        "best_epoch": int(train_result.best_epoch),
        "runtime_seconds": float(train_result.runtime_seconds),
        "checkpoint_path": str(train_result.checkpoint_path),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    """CLI entry point for submission runs."""
    args = parse_args()
    if args.repetitions < 5:
        raise ValueError("--repetitions must be at least 5 for official leaderboard reporting.")

    cfg = build_submission_cfg()
    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "submission_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(cfg.data.tdc_group_path))
    selected_datasets = resolve_datasets(args.datasets, group.dataset_names)

    print("Submission config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Selected datasets: {selected_datasets}")

    predictions_list: list[dict[str, np.ndarray]] = []

    for rep_idx, seed in enumerate(range(1, args.repetitions + 1), start=1):
        predictions: dict[str, np.ndarray] = {}

        for dataset_idx, dataset_name in enumerate(selected_datasets, start=1):
            print(
                f"Rep {rep_idx}/{args.repetitions} | "
                f"Dataset {dataset_idx}/{len(selected_datasets)}: {dataset_name} | "
                f"Seed {seed}"
            )

            run_dir = output_root / dataset_name / f"seed_{seed}"
            y_pred, train_result, task = run_one_training(
                cfg=cfg,
                dataset_name=dataset_name,
                seed=seed,
                output_dir=run_dir,
                group=group,
            )

            predictions[dataset_name] = y_pred
            save_run_artifacts(
                run_dir=run_dir,
                predictions=y_pred,
                train_result=train_result,
                dataset_name=dataset_name,
                task=task,
            )

        predictions_list.append(predictions)

        with (output_root / "predictions_list.pkl").open("wb") as f:
            pickle.dump(predictions_list, f)

    official_report = group.evaluate_many(predictions_list)
    (output_root / "official_report.json").write_text(json.dumps(official_report, indent=2))

    print("Official submission report:")
    print(json.dumps(official_report, indent=2))


if __name__ == "__main__":
    main()
