"""Paper-ready reporting utilities.

Provides:
    - save_mini_report(): writes mini_report.json with metrics, val curves,
      param counts, runtime, and config snapshot.
    - plot_training_curves(): NeurIPS-style loss/NLL/PICP curves (PDF).
    - plot_rejection_curve(): Rejection curve comparison between models (PDF).

All plots follow NeurIPS 2024 style: serif fonts, 10pt, single-column (3.5 in),
300 DPI raster, PDF as primary format with embedded fonts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed fonts in PDF
        "ps.fonttype": 42,
    }
)

_SINGLE_COL = 3.5
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


@dataclass
class MiniReport:
    """Structured report written to mini_report.json after each run."""

    dataset: str
    model: str
    task: str
    metrics: dict[str, float]
    val_curves: dict[str, list[float]]   # per-epoch NLL, PICP_95 on val set
    n_params_total: int
    n_params_trainable: int
    runtime_seconds: float
    config_snapshot: dict[str, Any]
    timestamp: str


def count_params(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: Any nn.Module.

    Returns:
        (total, trainable) parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_mini_report(
    *,
    dataset: str,
    model_name: str,
    task: str,
    metrics: dict[str, float],
    val_curves: dict[str, list[float]],
    model: nn.Module,
    runtime_seconds: float,
    config_snapshot: dict[str, Any],
    out_dir: str | Path,
) -> Path:
    """Serialise evaluation results to <out_dir>/mini_report.json.

    Args:
        dataset:          TDC dataset name.
        model_name:       "mlp" or "sv_dkl".
        task:             "regression" or "classification".
        metrics:          Final test metrics (flat dict, may include _std keys).
        val_curves:       Per-epoch calibration curves from TrainResult.val_metrics.
        model:            Trained nn.Module (for param counting).
        runtime_seconds:  Wall-clock training time.
        config_snapshot:  Resolved Hydra config as plain dict.
        out_dir:          Directory to write mini_report.json into.

    Returns:
        Path to the written JSON file.
    """
    total, trainable = count_params(model)
    report = MiniReport(
        dataset=dataset,
        model=model_name,
        task=task,
        metrics=metrics,
        val_curves=val_curves,
        n_params_total=total,
        n_params_trainable=trainable,
        runtime_seconds=round(runtime_seconds, 2),
        config_snapshot=config_snapshot,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "mini_report.json"
    path.write_text(json.dumps(asdict(report), indent=2))
    return path


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_metrics: dict[str, list[float]],
    out_path: str | Path,
    title: str = "",
) -> None:
    """Plot training/validation loss plus any extra per-epoch val metrics.

    Each entry in val_metrics gets its own subplot (e.g. NLL, PICP_95).

    Args:
        train_losses:  Per-epoch training loss.
        val_losses:    Per-epoch validation loss.
        val_metrics:   Extra per-epoch metrics, e.g. {"NLL": [...], "PICP_95": [...]}.
        out_path:      Output PDF path.
        title:         Optional figure suptitle.
    """
    n_extra = len(val_metrics)
    n_cols = 1 + n_extra
    fig, axes = plt.subplots(1, n_cols, figsize=(_SINGLE_COL * n_cols, 2.5))
    if n_cols == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    ax0 = axes[0]
    ax0.plot(epochs, train_losses, color=_PALETTE[0], label="Train", linewidth=1.2)
    ax0.plot(epochs, val_losses, color=_PALETTE[1], label="Val", linewidth=1.2, linestyle="--")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend(frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)

    for ax, (name, vals), colour in zip(axes[1:], val_metrics.items(), _PALETTE[2:]):
        ax.plot(range(1, len(vals) + 1), vals, color=colour, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_rejection_curve(
    results: dict[str, tuple[list[float], list[float]]],
    out_path: str | Path,
    ylabel: str = "MAE",
    title: str = "Rejection Curve",
) -> None:
    """Plot rejection curves for multiple models on the same axes.

    Args:
        results:  Dict mapping model_name -> (fractions, metrics) from
                  src.utils.metrics.rejection_curve().
        out_path: Output PDF path.
        ylabel:   Y-axis label (metric name).
        title:    Plot title.
    """
    fig, ax = plt.subplots(figsize=(_SINGLE_COL, 2.5))

    for (name, (fractions, metrics)), colour in zip(results.items(), _PALETTE):
        ax.plot(fractions, metrics, label=name, color=colour, linewidth=1.2)

    ax.set_xlabel("Fraction Rejected (highest uncertainty first)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
