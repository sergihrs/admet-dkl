"""XGBoost baseline: single-trial training and deep-ensemble helpers.

Stage 1 (Optuna sweep): ``train_single_xgb`` fits one model, returns predictions
and per-round train/val curves so ``main.py`` can compute a validation metric
for the sweeper.

Stage 2 (deep ensemble): ``train_single_xgb`` is looped N times with distinct
seeds derived via ``cantor_pairing``. Diversity across members comes from XGB's
row/column subsampling — those knobs MUST be tuned in stage 1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)


def cantor_pairing(a: int, b: int) -> int:
    """Bijective integer encoding of (a, b). Used to derive unique member seeds.

    Args:
        a: First integer (e.g. the base run seed).
        b: Second integer (e.g. the ensemble member index).

    Returns:
        Unique non-negative integer identifying the (a, b) pair.
    """
    return (a + b) * (a + b + 1) // 2 + b


def extract_arrays(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pull raw ``(X, y)`` numpy arrays from an in-memory DataLoader.

    Works for both a bare ``EmbeddingDataset`` (test split) and a
    ``torch.utils.data.Subset`` wrapping one (train/val splits).
    """
    ds = loader.dataset
    if isinstance(ds, Subset):
        base = ds.dataset
        idx = torch.as_tensor(ds.indices, dtype=torch.long)
        X_t = base.X[idx]  # type: ignore[attr-defined]
        y_t = base.y[idx]  # type: ignore[attr-defined]
    else:
        X_t = ds.X  # type: ignore[attr-defined]
        y_t = ds.y  # type: ignore[attr-defined]
    return X_t.numpy(), y_t.numpy()


@dataclass
class XGBSingleResult:
    """Output of a single XGBoost fit used by both stage 1 and stage 2."""

    model: Any
    train_curve: list[float]  # per-round eval metric on training set
    val_curve: list[float]  # per-round eval metric on validation set
    val_mu: np.ndarray  # validation predictions (probs for clf)
    test_mu: np.ndarray  # test predictions (probs for clf)
    n_trees: int
    runtime_seconds: float


def _build_estimator(cfg_model: DictConfig, task: str, seed: int) -> Any:
    """Instantiate an untrained XGB regressor/classifier from a model config."""
    common: dict[str, Any] = dict(
        n_estimators=int(cfg_model.n_estimators),
        learning_rate=float(cfg_model.learning_rate),
        max_depth=int(cfg_model.max_depth),
        subsample=float(cfg_model.subsample),
        colsample_bytree=float(cfg_model.colsample_bytree),
        min_child_weight=float(cfg_model.min_child_weight),
        gamma=float(cfg_model.gamma),
        reg_alpha=float(cfg_model.reg_alpha),
        reg_lambda=float(cfg_model.reg_lambda),
        tree_method=str(cfg_model.tree_method),
        device=str(cfg_model.device),
        random_state=int(seed),
        early_stopping_rounds=int(cfg_model.early_stopping_rounds),
    )
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", **common)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)


def train_single_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg_model: DictConfig,
    task: str,
    seed: int,
) -> XGBSingleResult:
    """Fit one XGBoost model with early stopping and collect predictions/history.

    Args:
        X_tr, y_tr: Training features/labels as numpy arrays.
        X_val, y_val: Validation features/labels (used for early stopping).
        X_test: Test features (predictions produced but not scored here).
        cfg_model: Hydra ``cfg.model`` subtree with XGB hyperparameters.
        task: "regression" or "classification".
        seed: Member seed. Combined with row/col subsampling this is what
              produces ensemble diversity across members.

    Returns:
        ``XGBSingleResult`` containing the fitted model, per-round curves,
        val/test predictions, tree count and wall-clock runtime.
    """
    model = _build_estimator(cfg_model, task, seed)
    t0 = time.perf_counter()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    runtime = time.perf_counter() - t0

    hist = model.evals_result()
    metric_key = next(iter(hist["validation_0"]))
    train_curve = [float(v) for v in hist["validation_0"][metric_key]]
    val_curve = [float(v) for v in hist["validation_1"][metric_key]]

    if task == "regression":
        val_mu = model.predict(X_val).astype(np.float32)
        test_mu = model.predict(X_test).astype(np.float32)
    else:
        val_mu = model.predict_proba(X_val)[:, 1].astype(np.float32)
        test_mu = model.predict_proba(X_test)[:, 1].astype(np.float32)

    return XGBSingleResult(
        model=model,
        train_curve=train_curve,
        val_curve=val_curve,
        val_mu=val_mu,
        test_mu=test_mu,
        n_trees=int(model.get_booster().num_boosted_rounds()),
        runtime_seconds=runtime,
    )
