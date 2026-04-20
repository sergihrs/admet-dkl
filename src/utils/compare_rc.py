"""Compare rejection curves stored in run mini reports.

This script reads two run folders (each containing ``mini_report.json``),
plots both test rejection curves on the same axes, and adds a random
baseline for visual comparison.

Usage examples:
    uv run python -m src.utils.compare_rc outputs/caco2_wang/xgboost/ensemble/2026-04-14_14-48-41 \
        outputs/caco2_wang/sv_dkl/sweeps/2026-04-11_17-53-17/10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rejection-curve comparison."""
    parser = argparse.ArgumentParser(description="Compare rejection curves from two run folders.")
    parser.add_argument(
        "run_a",
        type=str,
        help="First run folder path or folder name (must contain mini_report.json).",
    )
    parser.add_argument(
        "run_b",
        type=str,
        help="Second run folder path or folder name (must contain mini_report.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "rejection_curve_compare.pdf",
        help="Output PDF path for the comparison plot.",
    )
    return parser.parse_args()


def _resolve_run_dir(run_folder: str | Path, search_root: Path = Path("outputs")) -> Path:
    """Resolve a run folder from a direct path or a folder name."""
    candidate = Path(run_folder)
    if candidate.is_dir():
        return candidate

    # Allow using only the timestamp/leaf folder name for convenience.
    matches = [
        p.parent for p in search_root.rglob("mini_report.json") if p.parent.name == str(run_folder)
    ]
    if not matches:
        raise FileNotFoundError(f"Could not resolve run folder: {run_folder}")
    if len(matches) > 1:
        as_text = ", ".join(str(p) for p in matches)
        raise ValueError(f"Run folder name '{run_folder}' is ambiguous: {as_text}")
    return matches[0]


def _load_rejection_payload(run_folder: str | Path) -> tuple[str, str, list[float], list[float]]:
    """Load model name and test rejection-curve data from a mini report."""
    run_dir = _resolve_run_dir(run_folder)
    report_path = run_dir / "mini_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"mini_report.json not found in: {run_dir}")

    raw: dict[str, Any] = json.loads(report_path.read_text())

    if "model" not in raw:
        raise KeyError(f"Missing 'model' in {report_path}")
    if "rejection_curve_test" not in raw or raw["rejection_curve_test"] is None:
        raise KeyError(
            f"Missing 'rejection_curve_test' in {report_path}. Re-run with updated reporting code."
        )

    rejection = raw["rejection_curve_test"]
    if not isinstance(rejection, dict):
        raise ValueError(f"Invalid 'rejection_curve_test' type in {report_path}")

    x_raw = rejection.get("x")
    y_raw = rejection.get("y")
    if not isinstance(x_raw, list) or not isinstance(y_raw, list):
        raise KeyError(
            f"'rejection_curve_test' must contain list fields 'x' and 'y' in {report_path}"
        )
    if len(x_raw) != len(y_raw) or len(x_raw) == 0:
        raise ValueError(f"Invalid rejection curve lengths in {report_path}")

    x_vals = [float(v) for v in x_raw]
    y_vals = [float(v) for v in y_raw]
    metric = str(rejection.get("metric", "metric"))
    model_name = str(raw["model"])
    return model_name, metric, x_vals, y_vals


# Metrics with a known, clean optimum of 0 (lower = better). Curves for these
# can be extended to fraction=1.0 with value=0 without ambiguity.
_OPTIMUM_ZERO_METRICS: frozenset[str] = frozenset({"mae", "rmse", "error_rate"})


def _metric_label(metric_name: str) -> str:
    """Map stored metric identifiers to readable axis labels."""
    mapping = {
        "mae": "MAE",
        "rmse": "RMSE",
        "error_rate": "Error Rate",
    }
    return mapping.get(metric_name.lower(), metric_name)


def _extend_to_full(
    xs: list[float], ys: list[float], metric: str
) -> tuple[list[float], list[float]]:
    """Append (1.0, 0.0) endpoint when metric has a clean zero optimum."""
    if metric.lower() not in _OPTIMUM_ZERO_METRICS:
        return xs, ys
    if xs and xs[-1] >= 1.0:
        return xs, ys
    return xs + [1.0], ys + [0.0]


def compare_rejection_curves(
    run_a: str | Path,
    run_b: str | Path,
    out_path: str | Path | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    random_label: str = "random",
    axis_label_size: int = 11,
    legend_font_size: int = 10,
) -> Path:
    """Plot two run rejection curves plus a random baseline in NeurIPS format."""
    model_a, metric_a, x_a, y_a = _load_rejection_payload(run_a)
    model_b, metric_b, x_b, y_b = _load_rejection_payload(run_b)
    display_a = label_a if label_a is not None else model_a
    display_b = label_b if label_b is not None else model_b

    metric_name = metric_a if metric_a == metric_b else f"{metric_a} / {metric_b}"

    x_a, y_a = _extend_to_full(x_a, y_a, metric_a)
    x_b, y_b = _extend_to_full(x_b, y_b, metric_b)

    baseline_x = sorted(set(x_a + x_b))
    baseline_y = float(np.mean([y_a[0], y_b[0]]))

    # Calculate Area Under the Curve (AUC) using numpy's trapezoidal rule
    auc_a = np.trapz(y_a, x_a)
    auc_b = np.trapz(y_b, x_b)
    auc_base = np.trapz([baseline_y] * len(baseline_x), baseline_x)

    # NeurIPS Academic Styling Parameters
    neurips_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "axes.labelsize": axis_label_size,
        "font.size": 10,
        "legend.fontsize": legend_font_size,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(neurips_rc):
        # 5.5 inch width is standard for single-column figures in two-column layouts
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Okabe-Ito Colorblind Palette
        color_a = "#0072B2"  # Deep Blue
        color_b = "#D55E00"  # Vermilion
        color_base = "#7F7F7F"  # Gray

        # Plot A with shaded area
        ax.plot(x_a, y_a, linewidth=1.5, color=color_a, label=f"{display_a} (AUC: {auc_a:.3f})")
        ax.fill_between(x_a, y_a, alpha=0.15, color=color_a, edgecolor="none")

        # Plot B with shaded area
        ax.plot(x_b, y_b, linewidth=1.5, color=color_b, label=f"{display_b} (AUC: {auc_b:.3f})")
        ax.fill_between(x_b, y_b, alpha=0.15, color=color_b, edgecolor="none")

        # Plot Baseline (no shading, to avoid visual clutter)
        ax.plot(
            baseline_x,
            [baseline_y] * len(baseline_x),
            linestyle="--",
            linewidth=1.2,
            color=color_base,
            label=f"{random_label} (AUC: {auc_base:.3f})",
            alpha=0.8,
        )

        ax.set_xlabel("Fraction Rejected")
        ax.set_ylabel(_metric_label(metric_name))

        # Limit the axes so the shading looks flush against the bottom
        ax.set_xlim(left=0, right=1.0)
        ax.set_xticks(np.arange(0.0, 1.0 + 1e-9, 0.10))
        if min(y_a + y_b) >= 0:
            ax.set_ylim(bottom=0)

        # Remove top and right spines for a clean academic look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False, loc="best")

        out_file = (
            Path(out_path)
            if out_path is not None
            else Path("outputs") / f"rejection_curve_compare_{model_a}_vs_{model_b}.pdf"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Ensures labels aren't cut off when saving
        fig.tight_layout()

        fig.savefig(out_file, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return out_file


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output = compare_rejection_curves(args.run_a, args.run_b, args.out)
    print(f"Saved comparison plot to: {output}")


if __name__ == "__main__":
    main()
