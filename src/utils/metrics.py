"""Evaluation metrics for probabilistic and deterministic ADMET predictions.

Design contract:
    - TDC standard metrics (MAE, AUROC, etc.) are delegated to group.evaluate()
      and returned as a flat dict with upper-cased keys.
    - This module adds: NLL, PICP, Rejection Curve, and AURC.
    - All uncertainty functions accept (mu, sigma2) so they work identically
      for MC Dropout (empirical variance) and SV-DKL (GP posterior variance).
    - Metric std should be reported as std across multiple seeds, not bootstrap.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# TDC-delegated metrics — flattened to a simple {METRIC: value} dict
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {
    "roc-auc", "pr-auc", "spearman", "pcc", "r2",
    "f1", "accuracy", "precision", "recall",
    "micro-f1", "macro-f1", "kappa", "avg-roc-auc",
}


def evaluate_tdc(group: object, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Delegate evaluation to the TDC benchmark group; return a flat dict.

    Flattens the nested TDC result ``{dataset: {metric: val}}`` to
    ``{METRIC: val}`` so it can be merged cleanly with custom metrics.

    Args:
        group: An instantiated tdc.benchmark_group.admet_group object.
        predictions: Dict mapping dataset name to predicted-values array,
                     as expected by group.evaluate().

    Returns:
        Flat dict, e.g. ``{"MAE": 0.31}`` or ``{"ROC-AUC": 0.94}``.
    """
    nested: dict[str, dict[str, float]] = group.evaluate(predictions)  # type: ignore[attr-defined]
    flat: dict[str, float] = {}
    for ds_metrics in nested.values():
        for k, v in ds_metrics.items():
            flat[k.upper()] = float(v)
    return flat


def primary_metric_for_dataset(dataset: str, group_name: str = "admet_group") -> str:
    """Return the TDC primary metric name for a benchmark dataset.

    Examples: ``"caco2_wang" -> "mae"``, ``"half_life_obach" -> "spearman"``.
    """
    from tdc.metadata import bm_metric_names

    return str(bm_metric_names[group_name][dataset])


def tdc_evaluator_value(
    y_true: np.ndarray, y_pred: np.ndarray, metric_name: str
) -> float:
    """Score predictions with TDC's official Evaluator for a given metric."""
    from tdc import Evaluator

    return float(Evaluator(name=metric_name)(y_true, y_pred))


def higher_is_better(metric_name: str) -> bool:
    """True if the metric should be maximised (e.g. ROC-AUC, Spearman)."""
    return metric_name.lower() in _HIGHER_IS_BETTER


# ---------------------------------------------------------------------------
# Gaussian NLL  (regression)
# ---------------------------------------------------------------------------

def gaussian_nll(
    y_true: Tensor,
    mu: Tensor,
    sigma2: Tensor,
    eps: float = 1e-6,
) -> float:
    """Mean Gaussian negative log-likelihood.

    NLL = 0.5 * mean[ log(sigma^2) + (y - mu)^2 / sigma^2 ] + 0.5*log(2*pi)

    Args:
        y_true: Ground-truth labels [N].
        mu:     Predictive mean [N].
        sigma2: Predictive variance [N] (must be > 0).
        eps:    Small constant for numerical stability.

    Returns:
        Scalar mean NLL (lower is better).
    """
    sigma2 = sigma2.clamp(min=eps)
    nll = 0.5 * (sigma2.log() + (y_true - mu).pow(2) / sigma2 + math.log(2 * math.pi))
    return nll.mean().item()


# ---------------------------------------------------------------------------
# PICP — Prediction Interval Coverage Probability  (regression only)
# ---------------------------------------------------------------------------

def picp(
    y_true: Tensor,
    mu: Tensor,
    sigma2: Tensor,
    confidence: float = 0.95,
) -> float:
    """Prediction Interval Coverage Probability at a given confidence level.

    Under a Gaussian predictive distribution the (1-alpha) interval is
    mu ± z_{alpha/2} * sigma.

    Args:
        y_true:     Ground-truth labels [N].
        mu:         Predictive mean [N].
        sigma2:     Predictive variance [N].
        confidence: Nominal coverage, e.g. 0.95.

    Returns:
        Empirical coverage fraction (ideally == confidence).
    """
    from scipy.stats import norm  # type: ignore[import]

    z = norm.ppf((1 + confidence) / 2)  # e.g. 1.96 for 95%
    sigma = sigma2.clamp(min=1e-6).sqrt()
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = ((y_true >= lower) & (y_true <= upper)).float().mean().item()
    return covered


# ---------------------------------------------------------------------------
# Rejection Curve
# ---------------------------------------------------------------------------

def rejection_curve(
    y_true: Tensor,
    y_pred: Tensor,
    uncertainty: Tensor,
    metric_fn: str = "mae",
    n_bins: int = 20,
) -> tuple[list[float], list[float]]:
    """Compute a rejection curve: metric vs fraction of most-uncertain rejected.

    Args:
        y_true:      Ground-truth labels [N].
        y_pred:      Point predictions [N].
        uncertainty: Per-sample uncertainty scores [N].
        metric_fn:   "mae", "rmse", or "error_rate".
        n_bins:      Number of rejection thresholds to evaluate.

    Returns:
        fractions: List of rejection fractions [0, 1).
        metrics:   Corresponding metric value at each fraction.
    """
    order = torch.argsort(uncertainty, descending=True)
    N = len(y_true)
    fractions: list[float] = []
    metrics: list[float] = []

    for k in range(0, N, max(1, N // n_bins)):
        keep_idx = order[k:]
        if len(keep_idx) < 2:
            break

        yt = y_true[keep_idx]
        yp = y_pred[keep_idx]

        if metric_fn == "mae":
            val = (yt - yp).abs().mean().item()
        elif metric_fn == "rmse":
            val = (yt - yp).pow(2).mean().sqrt().item()
        elif metric_fn == "error_rate":
            val = ((yp > 0.5).float() != yt).float().mean().item()
        else:
            raise ValueError(f"Unknown metric_fn: {metric_fn}")

        fractions.append(k / N)
        metrics.append(val)

    return fractions, metrics


def area_under_rejection_curve(fractions: list[float], metrics: list[float]) -> float:
    """Trapezoid-rule AUC of the rejection curve (lower = better calibration).

    Args:
        fractions: Rejection fractions from rejection_curve().
        metrics:   Metric values from rejection_curve().

    Returns:
        Scalar AURC.
    """
    return float(np.trapz(metrics, fractions))
