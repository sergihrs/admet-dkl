"""Hydra multirun callback: records every trial and reports the top-5."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class BestJobsCallback(Callback):
    """Track all jobs in an Optuna sweep and emit the top-5 to disk.

    The metric is whatever the objective function returned. For regression
    sweeps that is val MAE, matching the key written to ``optimization_results.yaml``.
    """

    LIVE_FILENAME = "top5_live.txt"

    def __init__(self) -> None:
        self.direction: str | None = None
        self.output_dir: str | None = None
        self.jobs: list[tuple[int, float]] = []  # (job_num, metric)

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self.direction = config.hydra.sweeper.direction

    def _ranked_top5(self) -> list[tuple[int, float]]:
        reverse = self.direction == "maximize"
        return sorted(self.jobs, key=lambda x: x[1], reverse=reverse)[:5]

    def _write_live_top5(self) -> None:
        """Overwrite ``top5_live.txt`` so a killed sweep still leaves a winner."""
        if self.output_dir is None or not self.jobs:
            return
        top5 = self._ranked_top5()
        path = Path(self.output_dir) / self.LIVE_FILENAME
        lines = [
            f"# trials_completed: {len(self.jobs)}",
            f"# direction: {self.direction}",
            "rank  job_num  metric",
        ]
        lines.extend(f"{i + 1:>4}  {jn:>7}  {m:.6f}" for i, (jn, m) in enumerate(top5))
        path.write_text("\n".join(lines) + "\n")

    def on_job_end(self, config: DictConfig, job_return: Any, **kwargs: Any) -> None:
        self.output_dir = str(Path(job_return.hydra_cfg.hydra.runtime.output_dir).parent)
        job_num = int(job_return.hydra_cfg.hydra.job.num)
        metric = job_return._return_value
        if metric is None:
            return
        try:
            metric_f = float(metric)
        except (TypeError, ValueError):
            return
        if math.isnan(metric_f):
            return
        self.jobs.append((job_num, metric_f))
        self._write_live_top5()

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        if not self.jobs or self.output_dir is None:
            print("No successful jobs recorded.")
            return

        top5 = self._ranked_top5()
        best_job_num, best_metric = top5[0]

        print(f"Best job number: {best_job_num}  (MAE={best_metric:.6f})")
        with open(f"{self.output_dir}/optimization_results.yaml", "a") as f:
            f.write(f"best_job_num: {best_job_num}\n")
            f.write(f"best_MAE: {best_metric}\n")
            f.write("top_5_jobs:\n")
            for jn, m in top5:
                f.write(f"  - job_num: {jn}\n")
                f.write(f"    MAE: {m}\n")
