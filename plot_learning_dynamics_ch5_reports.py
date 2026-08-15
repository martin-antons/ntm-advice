#!/usr/bin/env python3
"""
Chapter 5.2 learning-dynamics plots for the NTM-with-Advice experiments.

This version uses the report statistics stored during training instead of the
raw single-example per-batch metrics. Each report point summarizes the
preceding report interval (200 training batches in the final experiments).
No additional moving average, interpolation, rolling mean, or post-hoc temporal
smoothing is applied.

For every figure:
  1. the same seed IDs are used for every advice condition,
  2. only the common available training horizon is shown,
  3. the plotted line is the mean across those seeds at each report point.

To keep multi-condition figures readable:
  - baseline_vs_combined plots show mean +/- one SD,
  - larger control/ablation plots show mean curves only.

The legend is placed below the coordinate system and no figure title is added.

Primary metrics:
  Copy:            mean BER over the stored report interval
  Even-Palindrome: mean classification accuracy over the report interval
  Repeat-Copy:     mean BER over the report interval

Loss is generated as a secondary metric by default.

Usage:
  python plot_learning_dynamics_ch5_reports.py
  python plot_learning_dynamics_ch5_reports.py --tasks copy
  python plot_learning_dynamics_ch5_reports.py --primary-only
  python plot_learning_dynamics_ch5_reports.py --log-loss
  python plot_learning_dynamics_ch5_reports.py --tasks repeat-copy --max-batch 50000

Output:
  analysis/ch5_learning_dynamics_reports/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Thesis figure definitions
# ---------------------------------------------------------------------------

GROUPS_BY_TASK = {
    "copy": {
        "baseline_vs_combined": (
            "none",
            "combined",
        ),
        "controls": (
            "none",
            "zero",
            "random",
            "wrong",
        ),
        "structural_features": (
            "none",
            "length",
            "position",
            "operation",
            "relation",
            "combined",
        ),
        "phase_ablation": (
            "none",
            "write_only",
            "read_only",
            "combined",
        ),
    },

    "even-palindrome": {
        "baseline_vs_combined": (
            "none",
            "combined",
        ),
        "controls": (
            "none",
            "zero",
            "random",
            "wrong",
        ),
        "structural_features": (
            "none",
            "length",
            "position",
            "operation",
            "relation",
            "combined",
        ),
        "comparison_structure": (
            "none",
            "position",
            "pair_index",
            "relation",
            "combined",
        ),
    },

    "repeat-copy": {
        "baseline_vs_combined": (
            "none",
            "combined",
        ),
        "controls": (
            "none",
            "zero",
            "random",
        ),
        "structural_features": (
            "none",
            "length",
            "position",
            "operation",
            "relation",
            "combined",
        ),
    },
}


PRIMARY_METRIC_BY_TASK = {
    "copy": "ber",
    "even-palindrome": "accuracy",
    "repeat-copy": "ber",
}

DEFAULT_METRICS_BY_TASK = {
    "copy": (
        "ber",
        "loss",
    ),
    "even-palindrome": (
        "accuracy",
        "loss",
    ),
    "repeat-copy": (
        "ber",
        "loss",
    ),
}


DISPLAY_NAMES = {
    "none": "None",
    "zero": "Zero",
    "random": "Random",
    "wrong": "Wrong",
    "length": "Length",
    "position": "Position",
    "operation": "Operation",
    "relation": "Relation",
    "combined": "Combined",
    "write_only": "Write only",
    "read_only": "Read only",
    "pair_index": "Pair index",
    "copy_only": "Copy only",
    "repeat_only": "Repeat only",
}


REPORT_METRIC_KEYS = {
    "loss": "mean_loss",
    "cost": "mean_cost",
    "ber": "mean_bit_error_rate",
    "accuracy": "mean_exact_sequence_accuracy",
}


# ---------------------------------------------------------------------------
# Run representation
# ---------------------------------------------------------------------------

@dataclass
class RunHistory:
    path: Path
    task: str
    advice: str
    seed: int
    status: str
    last_batch: int
    updated_at: str
    report_interval: int | None
    history: dict

    @property
    def key(self) -> Tuple[str, str, int]:
        return self.task, self.advice, self.seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Create Chapter 5.2 learning-dynamics plots from the stored "
            "training report means using common seeds and a common horizon."
        )
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "runs",
        help="Root searched recursively for history.json files.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis" / "ch5_learning_dynamics_reports",
        help="Output directory.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(GROUPS_BY_TASK.keys()),
        default=None,
        help="Default: all three tasks.",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Optional group filter, e.g. --groups controls structural_features. "
            "Unavailable groups for a task are ignored."
        ),
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(REPORT_METRIC_KEYS.keys()),
        default=None,
        help=(
            "Override task-specific default metrics. "
            "Example: --metrics loss ber accuracy"
        ),
    )

    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Generate only the primary task metric: BER for Copy/Repeat-Copy "
            "and classification accuracy for Even-Palindrome."
        ),
    )

    parser.add_argument(
        "--shade-all",
        action="store_true",
        help=(
            "Show mean +/- SD shading for every group. By default, shading "
            "is shown only for baseline_vs_combined."
        ),
    )

    parser.add_argument(
        "--no-shading",
        action="store_true",
        help="Disable SD shading in all figures.",
    )

    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help=(
            "Optional additional global cap. The automatic common horizon "
            "is still enforced."
        ),
    )

    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Optional lower batch bound. Default: 1.",
    )

    parser.add_argument(
        "--completed-only",
        action="store_true",
        help=(
            "Use only histories whose run status is 'completed'. "
            "Not recommended for the current Repeat-Copy analysis because "
            "interrupted histories contain valid training data."
        ),
    )

    parser.add_argument(
        "--log-loss",
        action="store_true",
        help="Use logarithmic y-axis for loss figures.",
    )

    parser.add_argument(
        "--pattern",
        default="history.json",
        help="Filename searched recursively. Default: history.json",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def normalize_task(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def normalize_advice(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def read_history(path: Path) -> RunHistory:
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    run = history.get("run", {})

    task = run.get(
        "task",
        history.get("task"),
    )

    advice = run.get(
        "advice_type",
        history.get(
            "model_params",
            {},
        ).get("advice_type", "none"),
    )

    seed = run.get(
        "seed",
        history.get("seed"),
    )

    if task is None or seed is None:
        raise ValueError(
            f"Missing task/seed metadata: {path}"
        )

    report_interval = run.get(
        "report_interval",
        history.get("training_args", {}).get("report_interval"),
    )
    if report_interval is not None:
        report_interval = int(report_interval)

    return RunHistory(
        path=path,
        task=normalize_task(task),
        advice=normalize_advice(advice),
        seed=int(seed),
        status=str(run.get("status", "unknown")),
        last_batch=int(run.get("last_batch", 0) or 0),
        updated_at=str(run.get("updated_at", "")),
        report_interval=report_interval,
        history=history,
    )


def discover_runs(
    root: Path,
    pattern: str,
) -> list[RunHistory]:

    if not root.exists():
        raise FileNotFoundError(
            f"Runs directory does not exist: {root}"
        )

    paths = sorted(
        root.rglob(pattern)
    )

    if not paths:
        raise FileNotFoundError(
            f"No {pattern!r} files found below {root}"
        )

    result = []

    for path in paths:
        try:
            result.append(
                read_history(path)
            )
        except Exception as exc:
            print(
                f"WARNING: skipping {path}: {exc}"
            )

    if not result:
        raise RuntimeError(
            "No readable histories found."
        )

    return result


def deduplicate_runs(
    runs: Iterable[RunHistory],
) -> list[RunHistory]:
    """
    If the same task/advice/seed appears more than once below runs/,
    prefer the longest history, then completed status, then latest update.
    """
    grouped = defaultdict(list)

    for run in runs:
        grouped[run.key].append(run)

    selected = []

    for key, candidates in grouped.items():
        preferred = max(
            candidates,
            key=lambda run: (
                run.last_batch,
                run.status == "completed",
                run.updated_at,
            ),
        )

        selected.append(
            preferred
        )

        if len(candidates) > 1:
            print(
                f"INFO: duplicate {key}; using {preferred.path}"
            )

    return sorted(
        selected,
        key=lambda run: (
            run.task,
            run.advice,
            run.seed,
        ),
    )


# ---------------------------------------------------------------------------
# Stored training-report metrics
# ---------------------------------------------------------------------------

def extract_report_metric(
    run: RunHistory,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:

    section = run.history.get(
        "reports",
        {},
    )

    batches = section.get(
        "batch",
        [],
    )

    values = section.get(
        REPORT_METRIC_KEYS[metric],
        [],
    )

    if not isinstance(batches, list) or not isinstance(values, list):
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )

    if len(batches) != len(values):
        raise ValueError(
            f"Length mismatch in {run.path}: "
            f"reports.batch={len(batches)}, "
            f"reports.{REPORT_METRIC_KEYS[metric]}={len(values)}"
        )

    clean_batch = []
    clean_value = []

    for batch, value in zip(
        batches,
        values,
    ):
        if value is None:
            continue

        batch = int(batch)
        value = float(value)

        if not np.isfinite(value):
            continue

        clean_batch.append(
            batch
        )
        clean_value.append(
            value
        )

    return (
        np.asarray(clean_batch, dtype=int),
        np.asarray(clean_value, dtype=float),
    )


def report_last_batch(
    run: RunHistory,
    metric: str,
) -> int:

    x, _ = extract_report_metric(
        run,
        metric,
    )

    if len(x) == 0:
        return 0

    return int(
        np.max(x)
    )


# ---------------------------------------------------------------------------
# Fair group selection
# ---------------------------------------------------------------------------

def select_group_runs(
    runs_by_advice: Mapping[str, Sequence[RunHistory]],
    requested_advices: Sequence[str],
    metric: str,
    global_max_batch: int | None,
):
    """
    Select the same seed IDs for every condition and determine a common
    per-batch training horizon.

    Returns:
        advices
        common_seeds
        selected_runs_by_advice
        common_horizon
    """

    advices = [
        advice
        for advice in requested_advices
        if advice in runs_by_advice
        and runs_by_advice[advice]
    ]

    if len(advices) < 2:
        return [], [], {}, 0

    seed_sets = []

    for advice in advices:
        usable_seeds = {
            run.seed
            for run in runs_by_advice[advice]
            if report_last_batch(run, metric) > 0
        }

        seed_sets.append(
            usable_seeds
        )

    if not seed_sets:
        return [], [], {}, 0

    common_seeds = sorted(
        set.intersection(
            *seed_sets
        )
    )

    if not common_seeds:
        return [], [], {}, 0

    selected = {}

    for advice in advices:
        by_seed = {
            run.seed: run
            for run in runs_by_advice[advice]
        }

        selected[advice] = [
            by_seed[seed]
            for seed in common_seeds
        ]

    horizons = []

    for advice in advices:
        for run in selected[advice]:
            horizons.append(
                report_last_batch(
                    run,
                    metric,
                )
            )

    common_horizon = min(
        horizons
    )

    if global_max_batch is not None:
        common_horizon = min(
            common_horizon,
            global_max_batch,
        )

    return (
        advices,
        common_seeds,
        selected,
        int(common_horizon),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_common_seeds(
    runs: Sequence[RunHistory],
    metric: str,
    common_seeds: Sequence[int],
    min_batch: int,
    max_batch: int,
):
    """
    Mean and SD across the same seed set at exactly the same report batch.

    Each stored report value already averages the preceding training-report
    interval. No additional temporal smoothing is performed here.
    """

    values_by_batch = defaultdict(
        dict
    )

    for run in runs:
        x, y = extract_report_metric(
            run,
            metric,
        )

        for batch, value in zip(
            x,
            y,
        ):
            batch = int(batch)

            if batch < min_batch:
                continue

            if batch > max_batch:
                continue

            values_by_batch[batch][
                run.seed
            ] = float(value)

    batches = []
    means = []
    stds = []
    minima = []
    maxima = []

    required = set(
        common_seeds
    )

    for batch in sorted(
        values_by_batch
    ):
        by_seed = values_by_batch[
            batch
        ]

        if set(by_seed) != required:
            continue

        values = np.asarray(
            [
                by_seed[seed]
                for seed in common_seeds
            ],
            dtype=float,
        )

        batches.append(
            batch
        )
        means.append(
            float(np.mean(values))
        )
        stds.append(
            float(np.std(values, ddof=0))
        )
        minima.append(
            float(np.min(values))
        )
        maxima.append(
            float(np.max(values))
        )

    return {
        "batch": np.asarray(
            batches,
            dtype=int,
        ),
        "mean": np.asarray(
            means,
            dtype=float,
        ),
        "std": np.asarray(
            stds,
            dtype=float,
        ),
        "min": np.asarray(
            minima,
            dtype=float,
        ),
        "max": np.asarray(
            maxima,
            dtype=float,
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def batch_formatter(
    value,
    _position,
):
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"

    return f"{int(value)}"


def y_label(
    task: str,
    metric: str,
) -> str:

    if metric == "loss":
        return "Mean loss"

    if metric == "ber":
        return "Mean bit error rate"

    if metric == "cost":
        return "Mean cost"

    if metric == "accuracy":
        if task == "even-palindrome":
            return "Mean classification accuracy"
        return "Mean exact sequence accuracy"

    return metric


def plot_group(
    *,
    task: str,
    group_name: str,
    requested_advices: Sequence[str],
    runs_by_advice: Mapping[str, Sequence[RunHistory]],
    metric: str,
    min_batch: int,
    global_max_batch: int | None,
    log_loss: bool,
    shade_all: bool,
    no_shading: bool,
    out_dir: Path,
):

    (
        advices,
        common_seeds,
        selected,
        common_horizon,
    ) = select_group_runs(
        runs_by_advice,
        requested_advices,
        metric,
        global_max_batch,
    )

    if len(advices) < 2:
        print(
            f"SKIP: {task}/{group_name}/{metric}: "
            "fewer than two usable conditions."
        )
        return None

    if not common_seeds:
        print(
            f"SKIP: {task}/{group_name}/{metric}: "
            "no common seeds."
        )
        return None

    if common_horizon < min_batch:
        print(
            f"SKIP: {task}/{group_name}/{metric}: "
            "no common report range."
        )
        return None

    print(
        f"PLOT: {task:16s} {group_name:24s} {metric:8s} "
        f"seeds={common_seeds} horizon={common_horizon}"
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.15)
    )

    csv_rows = []

    if no_shading:
        show_shading = False
    elif shade_all:
        show_shading = True
    else:
        show_shading = (group_name == "baseline_vs_combined")

    for advice in advices:
        aggregate = aggregate_common_seeds(
            selected[advice],
            metric,
            common_seeds,
            min_batch,
            common_horizon,
        )

        x = aggregate["batch"]
        mean = aggregate["mean"]
        std = aggregate["std"]
        minima = aggregate["min"]
        maxima = aggregate["max"]

        if len(x) == 0:
            continue

        line, = ax.plot(
            x,
            mean,
            linewidth=1.7,
            label=DISPLAY_NAMES.get(
                advice,
                advice,
            ),
            rasterized=True,
        )

        if show_shading:
            lower = mean - std
            upper = mean + std

            if metric in {"ber", "accuracy"}:
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

            ax.fill_between(
                x,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
                rasterized=True,
            )

        for (
            batch,
            mean_value,
            std_value,
            min_value,
            max_value,
        ) in zip(
            x,
            mean,
            std,
            minima,
            maxima,
        ):
            csv_rows.append(
                {
                    "task": task,
                    "group": group_name,
                    "metric": metric,
                    "advice_type": advice,
                    "batch": int(batch),
                    "mean": float(mean_value),
                    "std": float(std_value),
                    "min": float(min_value),
                    "max": float(max_value),
                    "n_seeds": len(common_seeds),
                    "seeds": ",".join(
                        str(seed)
                        for seed in common_seeds
                    ),
                    "common_horizon": common_horizon,
                    "report_interval": (
                        selected[advice][0].report_interval
                        if selected[advice]
                        else None
                    ),
                    "sd_shading": bool(show_shading),
                }
            )

    if not ax.lines:
        plt.close(
            fig
        )
        return None

    # No figure title.
    ax.set_xlabel(
        "Training batch"
    )

    ax.set_ylabel(
        y_label(
            task,
            metric,
        )
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            batch_formatter
        )
    )

    ax.set_xlim(
        max(0, min_batch),
        common_horizon,
    )

    if metric in {
        "ber",
        "accuracy",
    }:
        ax.set_ylim(
            -0.02,
            1.02,
        )

    if (
        metric == "loss"
        and log_loss
    ):
        ax.set_yscale(
            "log"
        )

    # Publication-style minimal axes.
    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.55,
    )

    # Legend below the coordinate system, never inside it.
    legend_columns = min(
        len(advices),
        3,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.20,
        ),
        ncol=legend_columns,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.97,
        bottom=0.27,
    )

    task_dir = (
        out_dir
        / task
    )

    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{group_name}_"
        f"{metric}"
    )

    pdf_path = (
        task_dir
        / f"{stem}.pdf"
    )

    png_path = (
        task_dir
        / f"{stem}.png"
    )

    csv_path = (
        task_dir
        / f"aggregated_{stem}.csv"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "group",
                "metric",
                "advice_type",
                "batch",
                "mean",
                "std",
                "min",
                "max",
                "n_seeds",
                "seeds",
                "common_horizon",
                "report_interval",
                "sd_shading",
            ],
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    return {
        "task": task,
        "group": group_name,
        "metric": metric,
        "advices": list(advices),
        "common_seeds": list(common_seeds),
        "n_seeds": len(common_seeds),
        "min_batch": min_batch,
        "common_horizon": common_horizon,
        "report_intervals": sorted({
            run.report_interval
            for advice in advices
            for run in selected[advice]
            if run.report_interval is not None
        }),
        "sd_shading": bool(show_shading),
        "pdf": str(pdf_path),
        "png": str(png_path),
        "csv": str(csv_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()

    tasks = (
        args.tasks
        if args.tasks
        else list(
            GROUPS_BY_TASK.keys()
        )
    )

    print("=" * 80)
    print("CHAPTER 5.2 LEARNING DYNAMICS -- REPORT MEANS")
    print("=" * 80)
    print(f"Runs root: {runs_dir}")
    print(f"Output:    {out_dir}")
    print("Source:    history['reports']")
    print("Temporal aggregation: stored report means only")
    print("Additional smoothing: none")
    print("Seeds:     intersection across all conditions in each figure")
    print("Horizon:   common available report range in each figure")
    print("Shading:   baseline-vs-combined only by default")
    print()

    discovered = discover_runs(
        runs_dir,
        args.pattern,
    )

    runs = deduplicate_runs(
        discovered
    )

    if args.completed_only:
        runs = [
            run
            for run in runs
            if run.status == "completed"
        ]

    runs = [
        run
        for run in runs
        if run.task in tasks
    ]

    if not runs:
        raise RuntimeError(
            "No usable runs remain."
        )

    runs_by_task = defaultdict(
        lambda: defaultdict(list)
    )

    for run in runs:
        runs_by_task[
            run.task
        ][
            run.advice
        ].append(
            run
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "runs_dir": str(runs_dir),
        "source": "reports",
        "additional_temporal_smoothing": False,
        "aggregation_description": (
            "Each report point is the stored mean over the training report "
            "interval; curves then average those report means across common "
            "seeds at identical report batches."
        ),
        "completed_only": bool(
            args.completed_only
        ),
        "global_max_batch": args.max_batch,
        "plots": [],
    }

    for task in tasks:
        if task not in runs_by_task:
            print(
                f"SKIP TASK: {task}: no runs found."
            )
            continue

        task_groups = GROUPS_BY_TASK[
            task
        ]

        if args.groups:
            requested_group_names = [
                group_name
                for group_name in args.groups
                if group_name in task_groups
            ]
        else:
            requested_group_names = list(
                task_groups.keys()
            )

        if args.primary_only:
            metrics = (
                PRIMARY_METRIC_BY_TASK[task],
            )
        elif args.metrics:
            metrics = tuple(args.metrics)
        else:
            metrics = DEFAULT_METRICS_BY_TASK[task]

        for group_name in requested_group_names:
            requested_advices = task_groups[
                group_name
            ]

            for metric in metrics:
                result = plot_group(
                    task=task,
                    group_name=group_name,
                    requested_advices=requested_advices,
                    runs_by_advice=runs_by_task[task],
                    metric=metric,
                    min_batch=args.min_batch,
                    global_max_batch=args.max_batch,
                    log_loss=args.log_loss,
                    shade_all=args.shade_all,
                    no_shading=args.no_shading,
                    out_dir=out_dir,
                )

                if result is not None:
                    manifest["plots"].append(
                        result
                    )

    manifest_path = (
        out_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print()
    print(
        f"Saved manifest: {manifest_path}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
