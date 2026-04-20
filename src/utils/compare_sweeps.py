"""Compare top-k runs from two sweeps and export a markdown summary.

Given two sweep directories, this script:
1. Finds the top-k runs from each sweep (default: 5).
2. Aggregates test metrics across those runs (mean and std).
3. Writes a markdown comparison report.
4. Plots rejection-curve comparison for the best run of each sweep.

Usage:
    uv run python compare_sweeps.py <sweep_a> <sweep_b>
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from src.utils.compare_rc import compare_rejection_curves


@dataclass(frozen=True)
class TopRun:
    """Container for a ranked run inside a sweep."""

    rank: int
    job_num: int
    objective_value: float | None


@dataclass(frozen=True)
class MetricSummary:
    """Mean/std summary for one metric."""

    mean_value: float
    std_value: float


@dataclass(frozen=True)
class SweepSummary:
    """Computed summary for one sweep."""

    sweep_path: Path
    use_spectral_norm: bool | None
    top_runs: list[TopRun]
    test_metrics: dict[str, MetricSummary]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare two sweep folders.")
    parser.add_argument("sweep_a", type=Path, help="Path to first sweep folder.")
    parser.add_argument("sweep_b", type=Path, help="Path to second sweep folder.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top runs to aggregate per sweep (default: 5).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to root/sweep_compare_<a>_vs_<b>.md",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help=("Output plot path. Defaults to root/sweep_compare_<a>_vs_<b>_rejection_curve.pdf"),
    )
    return parser.parse_args()


def _sanitize_fragment(value: str) -> str:
    """Return a filename-safe fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _default_outputs(sweep_a: Path, sweep_b: Path) -> tuple[Path, Path]:
    """Create default root-level output paths."""
    a_name = _sanitize_fragment(sweep_a.name)
    b_name = _sanitize_fragment(sweep_b.name)
    base = f"sweep_compare_{a_name}_vs_{b_name}"
    return Path(f"{base}.md"), Path(f"{base}_rejection_curve.pdf")


def _require_sweep_dir(path: Path) -> Path:
    """Validate that a path is a sweep directory."""
    if not path.is_dir():
        raise FileNotFoundError(f"Sweep path does not exist or is not a directory: {path}")
    return path


def _infer_spectral_norm(sweep_path: Path) -> bool | None:
    """Infer use_spectral_norm from run-0 Hydra config when available."""
    cfg_path = sweep_path / "0" / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None

    text = cfg_path.read_text(encoding="utf-8")
    match = re.search(r"use_spectral_norm:\s*(true|false)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _parse_top_runs_from_live(top5_path: Path, top_k: int) -> list[TopRun]:
    """Parse ranked top runs from top5_live.txt."""
    top_runs: list[TopRun] = []
    lines = top5_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        match = re.match(r"^\s*(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s*$", line)
        if match is None:
            continue

        rank = int(match.group(1))
        job_num = int(match.group(2))
        objective = float(match.group(3))
        top_runs.append(TopRun(rank=rank, job_num=job_num, objective_value=objective))

        if len(top_runs) >= top_k:
            break

    return top_runs


def _parse_top_runs_from_optimization_yaml(opt_path: Path, top_k: int) -> list[TopRun]:
    """Fallback parser for top runs in optimization_results.yaml."""
    lines = opt_path.read_text(encoding="utf-8").splitlines()
    in_top5 = False
    next_rank = 1
    top_runs: list[TopRun] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "top_5_jobs:":
            in_top5 = True
            continue
        if not in_top5:
            continue

        job_match = re.match(r"^-\s*job_num:\s*(\d+)\s*$", stripped)
        if job_match is not None:
            top_runs.append(
                TopRun(rank=next_rank, job_num=int(job_match.group(1)), objective_value=None)
            )
            next_rank += 1
            if len(top_runs) >= top_k:
                break

    return top_runs


def _load_top_runs(sweep_path: Path, top_k: int) -> list[TopRun]:
    """Load top-k run IDs using sweep summary artifacts."""
    top5_path = sweep_path / "top5_live.txt"
    if top5_path.exists():
        top_runs = _parse_top_runs_from_live(top5_path, top_k)
        if top_runs:
            return top_runs

    opt_path = sweep_path / "optimization_results.yaml"
    if opt_path.exists():
        top_runs = _parse_top_runs_from_optimization_yaml(opt_path, top_k)
        if top_runs:
            return top_runs

    raise RuntimeError(
        "Could not find top runs. Expected non-empty top5_live.txt or "
        f"optimization_results.yaml in: {sweep_path}"
    )


def _load_test_metrics(run_path: Path) -> dict[str, float]:
    """Load test metrics from one run mini report."""
    report_path = run_path / "mini_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"mini_report.json not found: {report_path}")

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    test_metrics_raw = raw.get("test_metrics")
    if not isinstance(test_metrics_raw, dict):
        raise KeyError(f"Missing or invalid 'test_metrics' in {report_path}")

    parsed: dict[str, float] = {}
    for metric_name, metric_value in test_metrics_raw.items():
        parsed[str(metric_name)] = float(metric_value)
    return parsed


def _summarize_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, MetricSummary]:
    """Compute mean and std for metrics available in all run dicts."""
    if not metric_dicts:
        raise ValueError("Cannot summarize metrics from an empty list.")

    shared_keys = list(metric_dicts[0].keys())
    for metric_dict in metric_dicts[1:]:
        shared_keys = [key for key in shared_keys if key in metric_dict]

    if not shared_keys:
        raise RuntimeError("No shared test metric keys found across the selected runs.")

    summary: dict[str, MetricSummary] = {}
    for key in shared_keys:
        values = [item[key] for item in metric_dicts]
        std_value = stdev(values) if len(values) > 1 else 0.0
        summary[key] = MetricSummary(mean_value=mean(values), std_value=std_value)

    return summary


def _build_sweep_summary(sweep_path: Path, top_k: int) -> SweepSummary:
    """Build a complete summary object for one sweep."""
    top_runs = _load_top_runs(sweep_path, top_k)
    metrics_per_run = [_load_test_metrics(sweep_path / str(entry.job_num)) for entry in top_runs]
    metric_summary = _summarize_metrics(metrics_per_run)
    use_spectral_norm = _infer_spectral_norm(sweep_path)

    return SweepSummary(
        sweep_path=sweep_path,
        use_spectral_norm=use_spectral_norm,
        top_runs=top_runs,
        test_metrics=metric_summary,
    )


def _spectral_label(value: bool | None) -> str:
    """Human-readable spectral norm status."""
    if value is True:
        return "with spectral norm"
    if value is False:
        return "without spectral norm"
    return "spectral norm unknown"


def _legend_label(value: bool | None, fallback: str) -> str:
    """Legend label for rejection-curve plot."""
    if value is True:
        return "with spectral norm"
    if value is False:
        return "without spectral norm"
    return fallback


def _format_summary_block(summary: SweepSummary) -> list[str]:
    """Render one sweep summary block in markdown."""
    top_jobs = ", ".join(str(item.job_num) for item in summary.top_runs)

    lines = [
        f"### Sweep: {summary.sweep_path}",
        f"- Setting: {_spectral_label(summary.use_spectral_norm)}",
        f"- Top {len(summary.top_runs)} jobs: {top_jobs}",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]

    for metric_name, metric_stats in summary.test_metrics.items():
        lines.append(
            f"| {metric_name} | {metric_stats.mean_value:.6f} | {metric_stats.std_value:.6f} |"
        )

    lines.append("")
    return lines


def _comparison_pair(
    summary_a: SweepSummary, summary_b: SweepSummary
) -> tuple[SweepSummary, SweepSummary, str]:
    """Choose comparison order and delta label."""
    if summary_a.use_spectral_norm is True and summary_b.use_spectral_norm is False:
        return summary_b, summary_a, "without_sn - with_sn"
    if summary_b.use_spectral_norm is True and summary_a.use_spectral_norm is False:
        return summary_a, summary_b, "without_sn - with_sn"
    return summary_b, summary_a, "sweep_b - sweep_a"


def _build_comparison_table(summary_a: SweepSummary, summary_b: SweepSummary) -> list[str]:
    """Create markdown table with metric deltas for means and stds."""
    minus_summary, plus_summary, delta_label = _comparison_pair(summary_a, summary_b)
    shared_metrics = [
        name for name in plus_summary.test_metrics if name in minus_summary.test_metrics
    ]

    lines = [
        f"## Comparison ({delta_label})",
        "| Metric | Delta Mean | Delta Std |",
        "|---|---:|---:|",
    ]

    for metric_name in shared_metrics:
        plus = plus_summary.test_metrics[metric_name]
        minus = minus_summary.test_metrics[metric_name]
        delta_mean = minus.mean_value - plus.mean_value
        delta_std = minus.std_value - plus.std_value
        lines.append(f"| {metric_name} | {delta_mean:+.6f} | {delta_std:+.6f} |")

    lines.append("")
    return lines


def _build_markdown(
    summary_a: SweepSummary,
    summary_b: SweepSummary,
    best_run_a: Path,
    best_run_b: Path,
    plot_path: Path,
) -> str:
    """Build the full markdown report text."""
    lines = [
        "# Sweep Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Best Run Rejection-Curve Comparison",
        f"- Sweep A best run: {best_run_a}",
        f"- Sweep B best run: {best_run_b}",
        f"- Plot path: {plot_path}",
        "",
        "## Top-K Test Metrics",
        "",
    ]

    lines.extend(_format_summary_block(summary_a))
    lines.extend(_format_summary_block(summary_b))
    lines.extend(_build_comparison_table(summary_a, summary_b))

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sweep_a = _require_sweep_dir(args.sweep_a)
    sweep_b = _require_sweep_dir(args.sweep_b)

    default_md, default_plot = _default_outputs(sweep_a, sweep_b)
    out_md: Path = args.out_md if args.out_md is not None else default_md
    out_plot: Path = args.out_plot if args.out_plot is not None else default_plot

    summary_a = _build_sweep_summary(sweep_a, args.top_k)
    summary_b = _build_sweep_summary(sweep_b, args.top_k)

    best_run_a = sweep_a / str(summary_a.top_runs[0].job_num)
    best_run_b = sweep_b / str(summary_b.top_runs[0].job_num)
    label_a = _legend_label(summary_a.use_spectral_norm, "sweep A")
    label_b = _legend_label(summary_b.use_spectral_norm, "sweep B")
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    actual_plot_path = compare_rejection_curves(
        best_run_a,
        best_run_b,
        out_plot,
        label_a=label_a,
        label_b=label_b,
        random_label="random",
    )

    report = _build_markdown(summary_a, summary_b, best_run_a, best_run_b, actual_plot_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")

    print(f"Saved markdown summary: {out_md}")
    print(f"Saved rejection-curve plot: {actual_plot_path}")


if __name__ == "__main__":
    main()
