#!/usr/bin/env python
"""Aggregate NTM training histories over random seeds and create plots.

The script supports both the new schema-v2 ``history.json`` and the older JSON
format with top-level ``losses``, ``costs`` and ``sequence_lengths`` lists.

Example:

    python aggregate_runs.py \
        --input-root ./runs/copy_advice_sweep \
        --output-dir ./plots/copy_advice_sweep \
        --window 200 \
        --band std \
        --show-individual
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunData:
    path: Path
    task: str
    advice_type: str
    seed: int
    status: str
    model_params: Dict[str, Any]
    batches: np.ndarray
    loss: np.ndarray
    cost: np.ndarray
    sequence_length: np.ndarray


IGNORED_COMPARABILITY_KEYS = {
    "name",
    "num_batches",
    "advice_type",
    "advice_strength",
    "advice_random_seed",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_history(path: Path) -> RunData:
    payload = load_json(path)

    if int(payload.get("schema_version", 1)) >= 2 and "metrics" in payload:
        run = payload.get("run", {})
        metrics = payload["metrics"]
        model_params = dict(payload.get("model_params", {}))
        batches = np.asarray(metrics.get("batch", []), dtype=np.int64)
        loss = np.asarray(metrics.get("loss", []), dtype=np.float64)
        cost = np.asarray(metrics.get("cost", []), dtype=np.float64)
        sequence_length = np.asarray(
            metrics.get("sequence_length", []),
            dtype=np.float64,
        )
        task = str(run.get("task", payload.get("task", "unknown")))
        advice_type = str(
            run.get(
                "advice_type",
                model_params.get("advice_type", "none"),
            )
        )
        seed = int(run.get("seed", payload.get("seed", -1)))
        status = str(run.get("status", "unknown"))
    else:
        # Backward compatibility with the original training JSON.
        model_params = dict(payload.get("model_params", {}))
        loss = np.asarray(payload.get("losses", []), dtype=np.float64)
        cost = np.asarray(payload.get("costs", []), dtype=np.float64)
        sequence_length = np.asarray(
            payload.get("sequence_lengths", []),
            dtype=np.float64,
        )
        batches = np.arange(1, len(loss) + 1, dtype=np.int64)
        task = str(payload.get("task", "unknown"))
        advice_type = str(model_params.get("advice_type", "none"))
        seed = int(payload.get("seed", -1))
        status = "completed"

    lengths = {len(batches), len(loss), len(cost), len(sequence_length)}
    if len(lengths) != 1:
        raise ValueError(
            "Metric arrays have different lengths in {}: {}".format(
                path,
                sorted(lengths),
            )
        )
    if len(loss) == 0:
        raise ValueError("No metric values in {}".format(path))
    if not np.isfinite(loss).all() or not np.isfinite(cost).all():
        raise ValueError("Non-finite loss/cost values in {}".format(path))

    return RunData(
        path=path,
        task=task,
        advice_type=advice_type,
        seed=seed,
        status=status,
        model_params=model_params,
        batches=batches,
        loss=loss,
        cost=cost,
        sequence_length=sequence_length,
    )


def discover_histories(root: Path) -> List[Path]:
    canonical = sorted(root.rglob("history.json"))
    if canonical:
        return canonical

    # Fallback for old folders containing only checkpoint-specific JSON files.
    candidates = sorted(root.rglob("*.json"))
    excluded_names = {
        "sweep_manifest.json",
        "aggregate.json",
        "sweep_copy_example.json",
    }
    candidates = [path for path in candidates if path.name not in excluded_names]

    # Keep only the latest checkpoint JSON for each (directory, task/advice/seed)
    # based on its stored batch_num.
    latest: Dict[Tuple[Path, str, str, int], Tuple[int, Path]] = {}
    for path in candidates:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "losses" not in payload:
            continue
        params = payload.get("model_params", {})
        key = (
            path.parent,
            str(payload.get("task", "unknown")),
            str(params.get("advice_type", "none")),
            int(payload.get("seed", -1)),
        )
        batch = int(payload.get("batch_num", len(payload.get("losses", []))))
        if key not in latest or batch > latest[key][0]:
            latest[key] = (batch, path)
    return sorted(value[1] for value in latest.values())


def block_reduce(
    batches: np.ndarray,
    values: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")

    x_values: List[int] = []
    y_values: List[float] = []
    for start in range(0, len(values), window):
        end = min(start + window, len(values))
        x_values.append(int(batches[end - 1]))
        y_values.append(float(values[start:end].mean()))
    return np.asarray(x_values), np.asarray(y_values)


def comparable_signature(params: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    items = []
    for key, value in sorted(params.items()):
        if key in IGNORED_COMPARABILITY_KEYS:
            continue
        items.append((str(key), json.dumps(value, sort_keys=True)))
    return tuple(items)


def check_comparability(runs: Sequence[RunData], strict: bool) -> None:
    signatures: Dict[Tuple[Tuple[str, str], ...], List[RunData]] = {}
    for run in runs:
        signatures.setdefault(comparable_signature(run.model_params), []).append(run)

    if len(signatures) <= 1:
        return

    description = []
    for index, grouped in enumerate(signatures.values(), start=1):
        example = grouped[0]
        description.append(
            "config {}: {} run(s), example={} advice={} seed={}".format(
                index,
                len(grouped),
                example.path,
                example.advice_type,
                example.seed,
            )
        )
    message = (
        "Histories contain different non-advice model configurations. "
        "Do not interpret them as one controlled comparison.\n"
        + "\n".join(description)
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message)


def confidence_width(stack: np.ndarray, band: str) -> np.ndarray:
    n = stack.shape[0]
    if n <= 1:
        return np.zeros(stack.shape[1], dtype=np.float64)
    std = stack.std(axis=0, ddof=1)
    if band == "std":
        return std
    if band == "sem":
        return std / np.sqrt(n)
    if band == "95ci":
        return 1.96 * std / np.sqrt(n)
    if band == "none":
        return np.zeros_like(std)
    raise ValueError("Unknown band {}".format(band))


def aggregate_group(
    runs: Sequence[RunData],
    metric: str,
    window: int,
    band: str,
) -> Dict[str, Any]:
    reduced = []
    for run in runs:
        values = getattr(run, metric)
        x, y = block_reduce(run.batches, values, window)
        reduced.append((run, x, y))

    # All runs are generated with one value per batch. For interrupted/legacy
    # runs, align to the shortest available curve instead of extrapolating.
    common_length = min(len(y) for _, _, y in reduced)
    x_reference = reduced[0][1][:common_length]
    stack = np.stack([y[:common_length] for _, _, y in reduced], axis=0)

    # Batch coordinates should be identical in a controlled sweep.
    for run, x, _ in reduced[1:]:
        if not np.array_equal(x[:common_length], x_reference):
            raise ValueError(
                "Incompatible batch coordinates between runs; first mismatch: {}".format(
                    run.path
                )
            )

    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    width = confidence_width(stack, band)

    return {
        "x": x_reference,
        "mean": mean,
        "std": std,
        "lower": mean - width,
        "upper": mean + width,
        "stack": stack,
        "seeds": [run.seed for run, _, _ in reduced],
        "paths": [str(run.path) for run, _, _ in reduced],
    }


def write_metric_csv(
    path: Path,
    task: str,
    metric: str,
    groups: Dict[str, Dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "metric",
                "advice_type",
                "batch",
                "n_seeds",
                "mean",
                "std",
                "lower_band",
                "upper_band",
            ],
        )
        writer.writeheader()
        for advice, aggregate in sorted(groups.items()):
            for index, batch in enumerate(aggregate["x"]):
                writer.writerow(
                    {
                        "task": task,
                        "metric": metric,
                        "advice_type": advice,
                        "batch": int(batch),
                        "n_seeds": len(aggregate["seeds"]),
                        "mean": float(aggregate["mean"][index]),
                        "std": float(aggregate["std"][index]),
                        "lower_band": float(aggregate["lower"][index]),
                        "upper_band": float(aggregate["upper"][index]),
                    }
                )


def write_final_summary(
    path: Path,
    task: str,
    grouped_runs: Dict[str, List[RunData]],
    window: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "advice_type",
                "n_seeds",
                "final_loss_mean",
                "final_loss_std",
                "final_cost_mean",
                "final_cost_std",
                "seeds",
            ],
        )
        writer.writeheader()
        for advice, runs in sorted(grouped_runs.items()):
            final_loss = np.asarray(
                [run.loss[-min(window, len(run.loss)):].mean() for run in runs]
            )
            final_cost = np.asarray(
                [run.cost[-min(window, len(run.cost)):].mean() for run in runs]
            )
            writer.writerow(
                {
                    "task": task,
                    "advice_type": advice,
                    "n_seeds": len(runs),
                    "final_loss_mean": float(final_loss.mean()),
                    "final_loss_std": float(final_loss.std(ddof=1)) if len(runs) > 1 else 0.0,
                    "final_cost_mean": float(final_cost.mean()),
                    "final_cost_std": float(final_cost.std(ddof=1)) if len(runs) > 1 else 0.0,
                    "seeds": ";".join(str(run.seed) for run in runs),
                }
            )


def plot_metric(
    path: Path,
    task: str,
    metric: str,
    groups: Dict[str, Dict[str, Any]],
    *,
    band: str,
    show_individual: bool,
    log_y: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    for advice, aggregate in sorted(groups.items()):
        x = aggregate["x"]
        mean = aggregate["mean"]
        line = axis.plot(
            x,
            mean,
            linewidth=2.2,
            label="{} (n={})".format(advice, len(aggregate["seeds"])),
        )[0]

        if band != "none" and len(aggregate["seeds"]) > 1:
            axis.fill_between(
                x,
                aggregate["lower"],
                aggregate["upper"],
                alpha=0.18,
                color=line.get_color(),
            )

        if show_individual:
            for seed_curve in aggregate["stack"]:
                axis.plot(
                    x,
                    seed_curve,
                    linewidth=0.8,
                    alpha=0.18,
                    color=line.get_color(),
                )

    axis.set_title("{} – {} across seeds".format(task, metric.capitalize()))
    axis.set_xlabel("Training batch")
    axis.set_ylabel(metric.capitalize())
    axis.grid(True, alpha=0.25)
    axis.legend()
    if log_y:
        axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def to_json_serializable(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for advice, aggregate in groups.items():
        result[advice] = {
            "batch": aggregate["x"].astype(int).tolist(),
            "mean": aggregate["mean"].tolist(),
            "std": aggregate["std"].tolist(),
            "lower": aggregate["lower"].tolist(),
            "upper": aggregate["upper"].tolist(),
            "seeds": aggregate["seeds"],
            "history_files": aggregate["paths"],
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default=None, help="Optional task filter")
    parser.add_argument(
        "--advices",
        nargs="*",
        default=None,
        help="Optional advice-type filter",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Non-overlapping averaging window in batches",
    )
    parser.add_argument(
        "--band",
        choices=["std", "sem", "95ci", "none"],
        default="std",
        help="Uncertainty band around the seed mean",
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Draw faint individual seed curves behind the mean",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include interrupted/running schema-v2 histories",
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Fail if non-advice hyperparameters differ across runs",
    )
    parser.add_argument(
        "--log-loss",
        action="store_true",
        help="Use a logarithmic y-axis for the loss plot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_paths = discover_histories(input_root)
    if not history_paths:
        raise FileNotFoundError("No training histories found below {}".format(input_root))

    runs = []
    for path in history_paths:
        try:
            run = parse_history(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            warnings.warn("Skipping {}: {}".format(path, error))
            continue

        if args.task is not None and run.task != args.task:
            continue
        if args.advices is not None and run.advice_type not in args.advices:
            continue
        if not args.include_incomplete and run.status not in {"completed", "unknown"}:
            continue
        runs.append(run)

    if not runs:
        raise ValueError("No histories remain after filtering")

    tasks = sorted({run.task for run in runs})
    if len(tasks) != 1:
        raise ValueError(
            "Aggregation expects one task at a time; found {}. Use --task.".format(tasks)
        )
    task = tasks[0]

    # Duplicate seeds within an advice group would overweight that seed.
    seen = set()
    for run in runs:
        key = (run.advice_type, run.seed)
        if key in seen:
            raise ValueError(
                "Duplicate advice/seed pair {} found. Separate experiments or remove old histories.".format(
                    key
                )
            )
        seen.add(key)

    check_comparability(runs, strict=args.strict_config)

    grouped_runs: Dict[str, List[RunData]] = {}
    for run in runs:
        grouped_runs.setdefault(run.advice_type, []).append(run)
    for grouped in grouped_runs.values():
        grouped.sort(key=lambda item: item.seed)

    aggregate_by_metric: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for metric in ("loss", "cost"):
        groups = {
            advice: aggregate_group(grouped, metric, args.window, args.band)
            for advice, grouped in grouped_runs.items()
        }
        aggregate_by_metric[metric] = groups
        write_metric_csv(
            output_dir / "{}_aggregate.csv".format(metric),
            task,
            metric,
            groups,
        )
        plot_metric(
            output_dir / "{}_mean_across_seeds.png".format(metric),
            task,
            metric,
            groups,
            band=args.band,
            show_individual=args.show_individual,
            log_y=args.log_loss if metric == "loss" else False,
        )

    write_final_summary(
        output_dir / "final_summary.csv",
        task,
        grouped_runs,
        args.window,
    )

    aggregate_json = {
        "task": task,
        "window": args.window,
        "band": args.band,
        "number_of_histories": len(runs),
        "groups": {
            metric: to_json_serializable(groups)
            for metric, groups in aggregate_by_metric.items()
        },
    }
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as file:
        json.dump(aggregate_json, file, indent=2, allow_nan=False)

    print("Aggregated {} runs for task '{}'.".format(len(runs), task))
    print("Outputs written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
