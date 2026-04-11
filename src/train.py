"""Training loops for MLP baseline and SV-DKL with validation-based early stopping.

Entry points:
    train_mlp()   — MSE/BCE loss, Adam, per-epoch NLL/PICP val tracking.
    train_dkl()   — GPyTorch PredictiveLogLikelihood ELBO, Adam loop, same tracking.

Both return a TrainResult that includes:
    - train/val loss histories
    - val_metrics curves:
        - "NLL" (both tasks)
        - "PICP_95" and "MAE" (regression)
        - "ROC_AUC" (classification)
    - best checkpoint path and wall-clock runtime

TensorBoard events are written to out_dir/tensorboard/ every epoch.
Launch with: tensorboard --logdir <sweep_or_run_dir>
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gpytorch
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.baseline_mlp import MLPPredictor
from src.models.sv_dkl import DKLModel, make_elbo
from src.utils.metrics import gaussian_nll, picp

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Outcome of a completed training run."""

    best_val_loss: float
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    # Per-epoch calibration curves tracked on the validation set.
    # Keys: "NLL" (both tasks), "PICP_95" and "MAE" (regression),
    # and "ROC_AUC" (classification).
    val_metrics: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    runtime_seconds: float = 0.0
    checkpoint_path: Path = Path(".")


class EarlyStopping:
    """Patience-based early stopping on validation loss.

    Args:
        patience:   Number of epochs without improvement before stopping.
        min_delta:  Minimum absolute improvement to count as progress.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0
        self.best_epoch: int = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self.best_epoch = epoch
        else:
            self._counter += 1
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Calibration-metric helpers (fast inference for per-epoch tracking)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _collect_mc_predictions(
    model: MLPPredictor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    T: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run MC Dropout on a DataLoader; return (mu, var, y) concatenated."""
    mu_list, var_list, y_list = [], [], []
    for X, y in loader:
        mu, var = model.predict_uncertainty(X.to(device), T=T)
        mu_list.append(mu.squeeze(-1).cpu())
        var_list.append(var.squeeze(-1).cpu())
        y_list.append(y)
    return torch.cat(mu_list), torch.cat(var_list), torch.cat(y_list)


@torch.no_grad()
def _collect_dkl_predictions(
    model: DKLModel,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Collect GP posterior predictions on a DataLoader; return (mu, var, y)."""
    mu_list, var_list, y_list = [], [], []
    with gpytorch.settings.fast_pred_var():
        for X, y in loader:
            mu, var = model.predict(X.to(device))
            mu_list.append(mu.cpu())
            var_list.append(var.cpu())
            y_list.append(y)
    return torch.cat(mu_list), torch.cat(var_list), torch.cat(y_list)


def _track_calibration_mlp(
    model: MLPPredictor,
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    task: str,
    device: torch.device,
    result: TrainResult,
    mc_val_samples: int = 10,
) -> None:
    """Append per-epoch NLL (and PICP for regression) to result.val_metrics.

    Uses T=mc_val_samples MC passes — fast enough to run every epoch.
    BatchNorm stays frozen (eval mode) during MC inference.
    """
    mu_v, var_v, y_v = _collect_mc_predictions(model, val_loader, T=mc_val_samples, device=device)

    if task == "regression":
        result.val_metrics.setdefault("NLL", []).append(gaussian_nll(y_v, mu_v, var_v))
        result.val_metrics.setdefault("PICP_95", []).append(picp(y_v, mu_v, var_v))
        result.val_metrics.setdefault("MAE", []).append((y_v - mu_v).abs().mean().item())
    else:
        # Binary NLL for classification (sigmoid not needed: predict_uncertainty
        # returns raw logits for MLP, so apply sigmoid here)
        p = torch.sigmoid(mu_v).clamp(1e-6, 1 - 1e-6)
        bin_nll = -(y_v * p.log() + (1 - y_v) * (1 - p).log()).mean().item()
        result.val_metrics.setdefault("NLL", []).append(bin_nll)
        try:
            auc = float(roc_auc_score(y_v.numpy(), p.numpy()))
        except ValueError:
            # Undefined when the validation split has a single class.
            auc = float("nan")
        result.val_metrics.setdefault("ROC_AUC", []).append(auc)


def _track_calibration_dkl(
    model: DKLModel,
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    result: TrainResult,
) -> None:
    """Append per-epoch NLL (and PICP for regression) to result.val_metrics."""
    mu_v, var_v, y_v = _collect_dkl_predictions(model, val_loader, device=device)

    if model.task == "regression":
        result.val_metrics.setdefault("NLL", []).append(gaussian_nll(y_v, mu_v, var_v))
        result.val_metrics.setdefault("PICP_95", []).append(picp(y_v, mu_v, var_v))
        result.val_metrics.setdefault("MAE", []).append((y_v - mu_v).abs().mean().item())
    else:
        # BernoulliLikelihood.predict() returns probabilities in [0,1]
        p = mu_v.clamp(1e-6, 1 - 1e-6)
        bin_nll = -(y_v * p.log() + (1 - y_v) * (1 - p).log()).mean().item()
        result.val_metrics.setdefault("NLL", []).append(bin_nll)
        try:
            auc = float(roc_auc_score(y_v.numpy(), p.numpy()))
        except ValueError:
            # Undefined when the validation split has a single class.
            auc = float("nan")
        result.val_metrics.setdefault("ROC_AUC", []).append(auc)


# ---------------------------------------------------------------------------
# MLP training loop
# ---------------------------------------------------------------------------


def _val_loss_mlp(
    model: MLPPredictor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate MLP on a split; returns mean criterion loss."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze(-1)
            total += criterion(pred, y).item() * len(y)
            count += len(y)
    return total / count


def train_mlp(
    model: MLPPredictor,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    cfg: DictConfig,
    task: str,
    out_dir: Path,
    device: torch.device,
) -> TrainResult:
    """Train the MLP baseline with Adam + ReduceLROnPlateau + early stopping.

    Per-epoch tracking: NLL and PICP_95 (regression) on the validation set
    using T=10 MC Dropout passes are stored in result.val_metrics.
    TensorBoard events are written to out_dir/tensorboard/.

    Args:
        model:        Instantiated MLPPredictor.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          Hydra train config (lr, max_epochs, patience, ...).
        task:         "regression" or "classification".
        out_dir:      Directory for checkpoint and TensorBoard logs.
        device:       Compute device.

    Returns:
        TrainResult with history and best checkpoint path.
    """
    model.to(device)
    criterion: nn.Module = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg.lr_patience, factor=0.5)
    stopper = EarlyStopping(patience=cfg.patience)

    ckpt_path = out_dir / "best_mlp.pt"
    result = TrainResult(best_val_loss=float("inf"), checkpoint_path=ckpt_path)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    t0 = time.perf_counter()

    for epoch in tqdm(range(1, cfg.max_epochs + 1), desc="MLP training"):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        val_loss = _val_loss_mlp(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Calibration curves on val set (T=10 MC passes for speed)
        _track_calibration_mlp(model, val_loader, task, device, result, mc_val_samples=10)
        if "NLL" in result.val_metrics:
            writer.add_scalar("Calibration/NLL", result.val_metrics["NLL"][-1], epoch)
        if "PICP_95" in result.val_metrics:
            writer.add_scalar("Calibration/PICP_95", result.val_metrics["PICP_95"][-1], epoch)
        if "MAE" in result.val_metrics:
            writer.add_scalar("Calibration/MAE", result.val_metrics["MAE"][-1], epoch)
        if "ROC_AUC" in result.val_metrics:
            writer.add_scalar("Calibration/ROC_AUC", result.val_metrics["ROC_AUC"][-1], epoch)

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        if stopper.step(val_loss, epoch):
            log.info("Early stopping at epoch %d (best epoch %d)", epoch, stopper.best_epoch)
            break

    writer.close()
    result.best_epoch = stopper.best_epoch
    result.runtime_seconds = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return result


# ---------------------------------------------------------------------------
# SV-DKL training loop
# ---------------------------------------------------------------------------


def train_dkl(
    model: DKLModel,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    cfg: DictConfig,
    n_train: int,
    out_dir: Path,
    device: torch.device,
) -> TrainResult:
    """Train SV-DKL with PredictiveLogLikelihood (ELBO) and early stopping.

    Validation loss uses MSE on the predictive mean (task-agnostic, comparable
    across seeds). Per-epoch NLL and PICP_95 are tracked separately via the
    GP posterior predictive distribution.
    TensorBoard events are written to out_dir/tensorboard/.

    Args:
        model:        Instantiated DKLModel.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          Hydra train config.
        n_train:      Total training set size (for KL scaling in ELBO).
        out_dir:      Directory for checkpoint and TensorBoard logs.
        device:       Compute device.

    Returns:
        TrainResult with history and best checkpoint path.
    """
    model.to(device)
    model.likelihood.to(device)

    mll = make_elbo(model, n_data=n_train)

    optimizer = Adam(
        [
            {
                "params": model.feature_extractor.parameters(),
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
            {"params": model.gp.parameters(), "lr": cfg.gp_lr, "weight_decay": 0.0},
            {"params": model.likelihood.parameters(), "lr": cfg.gp_lr, "weight_decay": 0.0},
        ],
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg.lr_patience, factor=0.5)
    stopper = EarlyStopping(patience=cfg.patience)

    ckpt_path = out_dir / "best_dkl.pt"
    result = TrainResult(best_val_loss=float("inf"), checkpoint_path=ckpt_path)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    t0 = time.perf_counter()

    for epoch in tqdm(range(1, cfg.max_epochs + 1), desc="DKL training"):
        model.train()
        model.likelihood.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = -mll(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= n_train

        # Val loss: MSE on predictive mean (for early stopping / LR schedule)
        model.eval()
        model.likelihood.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                mu, _ = model.predict(X)
                val_loss += nn.functional.mse_loss(mu, y, reduction="sum").item()
                val_count += len(y)
        val_loss /= val_count

        scheduler.step(val_loss)
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Calibration curves via GP posterior (model already in eval mode)
        _track_calibration_dkl(model, val_loader, device, result)
        if "NLL" in result.val_metrics:
            writer.add_scalar("Calibration/NLL", result.val_metrics["NLL"][-1], epoch)
        if "PICP_95" in result.val_metrics:
            writer.add_scalar("Calibration/PICP_95", result.val_metrics["PICP_95"][-1], epoch)
        if "MAE" in result.val_metrics:
            writer.add_scalar("Calibration/MAE", result.val_metrics["MAE"][-1], epoch)
        if "ROC_AUC" in result.val_metrics:
            writer.add_scalar("Calibration/ROC_AUC", result.val_metrics["ROC_AUC"][-1], epoch)

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(), "likelihood": model.likelihood.state_dict()},
                ckpt_path,
            )

        if stopper.step(val_loss, epoch):
            log.info("Early stopping at epoch %d", epoch)
            break

    writer.close()
    result.best_epoch = stopper.best_epoch
    result.runtime_seconds = time.perf_counter() - t0

    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.likelihood.load_state_dict(ckpt["likelihood"])
    return result
