#!/usr/bin/env python3
"""
Plot individual NTM training runs from recursively discovered history.json files.

This version supports several nested JSON layouts, including:
- list of records
- {"history": [records]}
- {"history": {"batch": [...], "loss": [...], "cost": [...]}}
- {"metrics": {"batch": [...], "loss": [...], "cost": [...]}}
- {"loss": [{"batch": 100, "value": ...}, ...], "cost": [...]}
- dictionaries keyed by batch number

Every run is plotted separately:
- same color for all seeds of one advice condition
- fixed line style per seed
- no mean curve and no standard-deviation band
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


CONDITION_COLORS = {
    "combined": "tab:blue",
    "length": "tab:orange",
    "none": "tab:green",
    "relation": "tab:red",
    "zero": "tab:purple",
    "random": "tab:brown",
    "position": "tab:pink",
    "operation": "tab:gray",
    "wrong": "tab:olive",
}

SEED_LINESTYLES = {
    1000: "-",
    1001: "--",
    1002: ":",
}

BATCH_KEYS = (
    "batch",
    "batches",
    "step",
    "steps",
    "iteration",
    "iterations",
    "train_batch",
    "batch_num",
    "batch_number",
)

METRIC_KEYS = {
    "loss": ("loss", "train_loss", "losses", "avg_loss", "mean_loss"),
    "cost": (
        "cost",
        "train_cost",
        "costs",
        "bit_error",
        "bit_errors",
        "error",
        "errors",
    ),
    "advice_weight_norm": (
        "advice_weight_norm",
        "advice_weights_norm",
        "weight_norm",
    ),
    "advice_gradient_norm": (
        "advice_gradient_norm",
        "advice_grad_norm",
        "gradient_norm",
        "grad_norm",
    ),
}

VALUE_KEYS = ("value", "values", "mean", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot every history.json run individually."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["loss", "cost"],
        choices=sorted(METRIC_KEYS),
    )
    parser.add_argument(
        "--smooth-points",
        type=int,
        default=1,
        help="Centered rolling-mean width in logged points. Default: 1.",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--legend-outside", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Print detected JSON structure for every history file.",
    )
    return parser.parse_args()


def first_present_key(mapping: dict[str, Any], candidates: Iterable[str]) -> str | None:
    lower_to_real = {str(key).lower(): key for key in mapping}
    for candidate in candidates:
        real = lower_to_real.get(candidate.lower())
        if real is not None:
            return real
    return None


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    valid = sum(to_float(item) is not None for item in value)
    return valid >= max(1, int(0.8 * len(value)))


def iter_nodes(node: Any, path: tuple[str, ...] = ()):
    """Yield every nested JSON node together with its JSON path."""
    yield path, node

    if isinstance(node, dict):
        for key, value in node.items():
            yield from iter_nodes(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            # Do not make debug paths excessively large.
            yield from iter_nodes(value, path + (f"[{index}]",))


def records_to_series(
    records: list[dict[str, Any]],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not records:
        return None

    batch_key = None
    metric_key = None

    for record in records:
        if batch_key is None:
            batch_key = first_present_key(record, BATCH_KEYS)
        if metric_key is None:
            metric_key = first_present_key(record, METRIC_KEYS[metric_name])
        if batch_key is not None and metric_key is not None:
            break

    if batch_key is None or metric_key is None:
        return None

    batches: list[float] = []
    values: list[float] = []

    for record in records:
        batch = to_float(record.get(batch_key))
        value = to_float(record.get(metric_key))
        if batch is not None and value is not None:
            batches.append(batch)
            values.append(value)

    if not batches:
        return None

    order = np.argsort(np.asarray(batches))
    return (
        np.asarray(batches, dtype=float)[order],
        np.asarray(values, dtype=float)[order],
    )


def parallel_arrays_to_series(
    mapping: dict[str, Any],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    batch_key = first_present_key(mapping, BATCH_KEYS)
    metric_key = first_present_key(mapping, METRIC_KEYS[metric_name])

    if batch_key is None or metric_key is None:
        return None

    batches_raw = mapping.get(batch_key)
    values_raw = mapping.get(metric_key)

    if not is_numeric_sequence(batches_raw) or not is_numeric_sequence(values_raw):
        return None

    count = min(len(batches_raw), len(values_raw))
    pairs = [
        (to_float(batches_raw[index]), to_float(values_raw[index]))
        for index in range(count)
    ]
    pairs = [(batch, value) for batch, value in pairs if batch is not None and value is not None]

    if not pairs:
        return None

    batches, values = zip(*pairs)
    order = np.argsort(np.asarray(batches))
    return (
        np.asarray(batches, dtype=float)[order],
        np.asarray(values, dtype=float)[order],
    )


def metric_specific_list_to_series(
    mapping: dict[str, Any],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Support layouts such as:
        {"loss": [{"batch": 100, "value": 0.6}, ...]}
    """
    metric_key = first_present_key(mapping, METRIC_KEYS[metric_name])
    if metric_key is None:
        return None

    metric_data = mapping.get(metric_key)
    if not isinstance(metric_data, list) or not metric_data:
        return None

    if all(isinstance(item, dict) for item in metric_data):
        batches: list[float] = []
        values: list[float] = []

        for item in metric_data:
            batch_key = first_present_key(item, BATCH_KEYS)
            value_key = first_present_key(
                item,
                METRIC_KEYS[metric_name] + VALUE_KEYS,
            )
            if batch_key is None or value_key is None:
                continue

            batch = to_float(item.get(batch_key))
            value = to_float(item.get(value_key))
            if batch is not None and value is not None:
                batches.append(batch)
                values.append(value)

        if batches:
            order = np.argsort(np.asarray(batches))
            return (
                np.asarray(batches, dtype=float)[order],
                np.asarray(values, dtype=float)[order],
            )

    return None


def batch_keyed_dict_to_series(
    mapping: dict[str, Any],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Support layouts such as:
        {
            "100": {"loss": 0.6, "cost": 35},
            "200": {"loss": 0.5, "cost": 30}
        }
    """
    batches: list[float] = []
    values: list[float] = []

    for key, value in mapping.items():
        batch = to_float(key)
        if batch is None or not isinstance(value, dict):
            continue

        metric_key = first_present_key(value, METRIC_KEYS[metric_name])
        if metric_key is None:
            continue

        metric_value = to_float(value.get(metric_key))
        if metric_value is not None:
            batches.append(batch)
            values.append(metric_value)

    if not batches:
        return None

    order = np.argsort(np.asarray(batches))
    return (
        np.asarray(batches, dtype=float)[order],
        np.asarray(values, dtype=float)[order],
    )


def implicit_batch_series(
    mapping: dict[str, Any],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Last-resort support for:
        {"loss": [0.6, 0.5, ...]}
    without explicit batch numbers.

    In that case use indices 1..N. This should rarely be needed.
    """
    metric_key = first_present_key(mapping, METRIC_KEYS[metric_name])
    if metric_key is None:
        return None

    values_raw = mapping.get(metric_key)
    if not is_numeric_sequence(values_raw):
        return None

    values = np.asarray(
        [to_float(value) for value in values_raw if to_float(value) is not None],
        dtype=float,
    )
    if values.size == 0:
        return None

    batches = np.arange(1, len(values) + 1, dtype=float)
    return batches, values


def extract_series(
    history_path: Path,
    metric_name: str,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    with history_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if debug:
        if isinstance(data, dict):
            print(
                f"[debug] {history_path}: top-level dict keys = "
                f"{list(data.keys())[:20]}"
            )
        elif isinstance(data, list):
            print(f"[debug] {history_path}: top-level list, length={len(data)}")
        else:
            print(f"[debug] {history_path}: top-level type={type(data).__name__}")

    # Search every nested node. Prefer explicit batch + metric layouts.
    for json_path, node in iter_nodes(data):
        if isinstance(node, list) and node and all(isinstance(item, dict) for item in node):
            result = records_to_series(node, metric_name)
            if result is not None:
                if debug:
                    print(f"[debug] detected record list at {'/'.join(json_path) or '<root>'}")
                return result

        if isinstance(node, dict):
            result = parallel_arrays_to_series(node, metric_name)
            if result is not None:
                if debug:
                    print(f"[debug] detected parallel arrays at {'/'.join(json_path) or '<root>'}")
                return result

            result = metric_specific_list_to_series(node, metric_name)
            if result is not None:
                if debug:
                    print(f"[debug] detected metric record list at {'/'.join(json_path) or '<root>'}")
                return result

            result = batch_keyed_dict_to_series(node, metric_name)
            if result is not None:
                if debug:
                    print(f"[debug] detected batch-keyed dict at {'/'.join(json_path) or '<root>'}")
                return result

    # Last resort: metric values without explicit batch axis.
    for json_path, node in iter_nodes(data):
        if isinstance(node, dict):
            result = implicit_batch_series(node, metric_name)
            if result is not None:
                if debug:
                    print(
                        f"[debug] detected implicit indexed series at "
                        f"{'/'.join(json_path) or '<root>'}"
                    )
                return result

    top_level_description = (
        f"dict keys={list(data.keys())[:20]}"
        if isinstance(data, dict)
        else f"type={type(data).__name__}"
    )
    raise ValueError(
        f"Could not detect metric {metric_name!r}; {top_level_description}. "
        "Run again with --debug-json to inspect the structure."
    )


def infer_seed(path: Path) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_condition(path: Path) -> str:
    lower_parts = [part.lower() for part in path.parts]

    for condition in CONDITION_COLORS:
        if condition in lower_parts:
            return condition

    for index, part in enumerate(lower_parts):
        if re.fullmatch(r"seed[_-]?\d+", part) and index > 0:
            return lower_parts[index - 1]

    return path.parent.parent.name.lower()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return values

    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(
        values,
        (window // 2, window - 1 - window // 2),
        mode="edge",
    )
    return np.convolve(padded, kernel, mode="valid")


def color_for_condition(condition: str, fallback_index: int) -> str:
    return CONDITION_COLORS.get(condition, f"C{fallback_index % 10}")


def linestyle_for_seed(seed: int | None, seed_rank: int) -> str:
    if seed in SEED_LINESTYLES:
        return SEED_LINESTYLES[seed]
    return ["-", "--", ":", "-."][seed_rank % 4]


def discover_histories(input_root: Path) -> list[Path]:
    histories = sorted(input_root.rglob("history.json"))
    if not histories:
        raise FileNotFoundError(
            f"No history.json files found below: {input_root.resolve()}"
        )
    return histories


def plot_metric(
    histories: list[Path],
    metric: str,
    args: argparse.Namespace,
) -> None:
    selected_conditions = (
        {condition.lower() for condition in args.conditions}
        if args.conditions
        else None
    )

    run_entries: list[dict[str, Any]] = []
    skipped: list[str] = []

    for history_path in histories:
        condition = infer_condition(history_path)
        if selected_conditions is not None and condition not in selected_conditions:
            continue

        seed = infer_seed(history_path)

        try:
            batches, values = extract_series(
                history_path,
                metric,
                debug=args.debug_json,
            )
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            skipped.append(f"{history_path}: {exc}")
            continue

        if args.max_batch is not None:
            mask = batches <= args.max_batch
            batches = batches[mask]
            values = values[mask]

        if batches.size == 0:
            skipped.append(f"{history_path}: no points remain after filtering")
            continue

        values = rolling_mean(values, args.smooth_points)

        if args.log_y:
            values = np.maximum(values, 1e-8)

        run_entries.append(
            {
                "path": history_path,
                "condition": condition,
                "seed": seed,
                "batches": batches,
                "values": values,
            }
        )

    if not run_entries:
        print(f"[warning] No usable runs found for metric {metric!r}.", file=sys.stderr)
        for message in skipped:
            print(f"  - {message}", file=sys.stderr)
        return

    condition_order = [
        condition
        for condition in CONDITION_COLORS
        if any(entry["condition"] == condition for entry in run_entries)
    ]
    condition_order.extend(
        sorted(
            {
                entry["condition"]
                for entry in run_entries
                if entry["condition"] not in CONDITION_COLORS
            }
        )
    )
    condition_rank = {condition: index for index, condition in enumerate(condition_order)}

    seed_values = sorted(
        {
            entry["seed"]
            for entry in run_entries
            if entry["seed"] is not None
        }
    )
    seed_rank = {seed: index for index, seed in enumerate(seed_values)}

    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    for entry in sorted(
        run_entries,
        key=lambda item: (
            condition_rank[item["condition"]],
            item["seed"] if item["seed"] is not None else 10**12,
        ),
    ):
        condition = entry["condition"]
        seed = entry["seed"]
        seed_label = f"seed {seed}" if seed is not None else entry["path"].parent.name

        ax.plot(
            entry["batches"],
            entry["values"],
            color=color_for_condition(condition, condition_rank[condition]),
            linestyle=linestyle_for_seed(seed, seed_rank.get(seed, 0)),
            linewidth=1.8,
            alpha=0.9,
            label=f"{condition} — {seed_label}",
        )

    task_name = args.task or args.input_root.name
    metric_title = metric.replace("_", " ").title()

    ax.set_title(f"{task_name} – Individual {metric_title} Runs")
    ax.set_xlabel("Training batch")
    ax.set_ylabel(metric_title)
    ax.grid(True, alpha=0.25)

    if metric in {"loss", "cost"} and not args.log_y:
        ax.set_ylim(bottom=0)

    if args.log_y:
        ax.set_yscale("log")

    if args.max_batch is not None:
        ax.set_xlim(right=args.max_batch)

    if args.legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 0.80, 1))
    else:
        ax.legend(loc="best", fontsize=9, ncol=2)
        fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if args.log_y else ""
    stem = f"{task_name}_{metric}_individual_runs{suffix}"

    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[info] Plotted {len(run_entries)} runs for {metric}.")

    if skipped:
        print(f"[warning] Skipped {len(skipped)} histories for {metric}:")
        for message in skipped:
            print(f"  - {message}")


def main() -> None:
    args = parse_args()

    if args.smooth_points < 1:
        raise SystemExit("--smooth-points must be at least 1.")

    histories = discover_histories(args.input_root)
    print(f"[info] Found {len(histories)} history.json files.")

    for metric in args.metrics:
        plot_metric(histories, metric, args)


if __name__ == "__main__":
    main()