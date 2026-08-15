#!/usr/bin/env python3
"""Quantitative Chapter 5.3 generalization evaluation.

Compares `none` and `combined` at fixed checkpoints on deterministic test sets.

Defaults
--------
Copy:
    checkpoint 20,000
    n = 10, 20, 30, 50, 100
    500 examples per n
    primary metric: BER

Even-Palindrome:
    checkpoint 100,000
    n = 10, 20, 30, 50, 100
    500 examples per n
    primary metric: classification accuracy

Repeat-Copy:
    checkpoint 50,000
    n = 10, 20, 30, 50
    r = 1, 3, 5, 7, 10
    200 examples per (n, r)
    primary metric: BER

All models of the same task/test condition see exactly the same test examples,
which are generated from a dedicated evaluation seed. Test data are therefore
separate from training and validation data.

The script uses fixed periodic checkpoints rather than best-validation
checkpoints so compared models received the same amount of training.

Output
------
analysis/ch5_generalization/
    copy_generalization_ber.pdf/.png
    copy_per_model.csv
    copy_aggregate.csv

    even-palindrome_generalization_accuracy.pdf/.png
    even-palindrome_per_model.csv
    even-palindrome_aggregate.csv

    repeat-copy_none_ber_heatmap.pdf/.png
    repeat-copy_combined_ber_heatmap.pdf/.png
    repeat-copy_per_model.csv
    repeat-copy_aggregate.csv
    repeat-copy_difference_combined_minus_none.csv

    manifest.json

Expected repository layout
--------------------------
Place this file next to train.py. By default it reads from:
    runs/final/<task>/<advice>/seed_<seed>/checkpoint-batch-XXXXXXXX.pt

Usage
-----
    python generalization_ch5.py
    python generalization_ch5.py --device cuda
    python generalization_ch5.py --tasks copy
    python generalization_ch5.py --tasks repeat-copy
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import attr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from train import evaluate
from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.evenpalindrometask import (
    EvenPalindromeTaskModelTraining,
    EvenPalindromeTaskParams,
)
from tasks.repeatcopytask import (
    RepeatCopyTaskModelTraining,
    RepeatCopyTaskParams,
)


ADVICES = ("none", "combined")
TEST_SEEDS = {
    "copy": (1000, 1001, 1002, 1003, 1004, 1005),
    "even-palindrome": (1000, 1001, 1002, 1003, 1004, 1005),
    "repeat-copy": (1000, 1001, 1002, 1003),
}
DEFAULT_CHECKPOINTS = {
    "copy": 20_000,
    "even-palindrome": 100_000,
    "repeat-copy": 50_000,
}
DEFAULT_LENGTHS = {
    "copy": (10, 20, 30, 50, 100),
    "even-palindrome": (10, 20, 30, 50, 100),
    "repeat-copy": (10, 20, 30, 50),
}
DEFAULT_REPETITIONS = (1, 3, 5, 7, 10)
DEFAULT_TEST_SEED = 20260815

TASK_CLASSES = {
    "copy": (CopyTaskModelTraining, CopyTaskParams),
    "even-palindrome": (
        EvenPalindromeTaskModelTraining,
        EvenPalindromeTaskParams,
    ),
    "repeat-copy": (
        RepeatCopyTaskModelTraining,
        RepeatCopyTaskParams,
    ),
}

CHECKPOINT_RE = re.compile(r"checkpoint-batch-(\d+)\.pt$")


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Quantitative OOD generalization analysis for Chapter 5.3."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "runs" ,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis" / "ch5_generalization",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_CLASSES),
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=20)

    parser.add_argument("--copy-examples", type=int, default=500)
    parser.add_argument("--palindrome-examples", type=int, default=500)
    parser.add_argument("--repeat-examples", type=int, default=200)

    parser.add_argument(
        "--copy-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS["copy"]),
    )
    parser.add_argument(
        "--palindrome-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS["even-palindrome"]),
    )
    parser.add_argument(
        "--repeat-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS["repeat-copy"]),
    )
    parser.add_argument(
        "--repeat-repetitions",
        nargs="+",
        type=int,
        default=list(DEFAULT_REPETITIONS),
    )

    parser.add_argument(
        "--copy-checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINTS["copy"],
    )
    parser.add_argument(
        "--palindrome-checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINTS["even-palindrome"],
    )
    parser.add_argument(
        "--repeat-checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINTS["repeat-copy"],
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable.")
    return torch.device(name)


def find_checkpoint(seed_dir: Path, batch_num: int) -> Path:
    canonical = seed_dir / f"checkpoint-batch-{int(batch_num):08d}.pt"
    if canonical.exists():
        return canonical

    matches = []
    if seed_dir.exists():
        for path in seed_dir.glob("checkpoint-batch-*.pt"):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match and int(match.group(1)) == int(batch_num):
                matches.append(path)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoints for batch {batch_num} in {seed_dir}"
        )
    raise FileNotFoundError(
        f"Missing checkpoint for batch {batch_num}. Expected: {canonical}"
    )


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_base_model(task: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load(checkpoint_path, device)
    checkpoint_task = str(checkpoint.get("task", "")).lower().replace("_", "-")
    if checkpoint_task != task:
        raise ValueError(
            f"{checkpoint_path}: task={checkpoint_task!r}, expected {task!r}"
        )

    model_cls, params_cls = TASK_CLASSES[task]
    params = params_cls(**checkpoint["model_params"])
    model = model_cls(params=params)
    model.net.load_state_dict(checkpoint["model_state_dict"])
    model.net.to(device)
    model.net.eval()

    metadata = {
        "seed": int(checkpoint.get("seed", -1)),
        "batch_num": int(checkpoint["batch_num"]),
        "advice_type": str(params.advice_type),
    }
    return model, metadata


def make_test_model(task: str, base_model, sequence_length: int, batch_size: int):
    params = attr.evolve(
        base_model.params,
        sequence_min_len=int(sequence_length),
        sequence_max_len=int(sequence_length),
        batch_size=int(batch_size),
    )
    model_cls = TASK_CLASSES[task][0]
    model = model_cls(params=params)
    model.net.load_state_dict(base_model.net.state_dict())
    device = next(base_model.net.parameters()).device
    model.net.to(device)
    model.net.eval()
    return model


def fixed_condition_seed(
    base_seed: int,
    task: str,
    sequence_length: int,
    repetitions: int | None = None,
) -> int:
    task_offset = {
        "copy": 1_000_000,
        "even-palindrome": 2_000_000,
        "repeat-copy": 3_000_000,
    }[task]
    value = int(base_seed) + task_offset + 10_007 * int(sequence_length)
    if repetitions is not None:
        value += 1_009 * int(repetitions)
    return value


def validate_example_count(name: str, n: int, batch_size: int):
    if n <= 0 or batch_size <= 0:
        raise ValueError(f"{name} and --eval-batch-size must be positive.")
    if n % batch_size != 0:
        raise ValueError(
            f"{name}={n} must be divisible by --eval-batch-size={batch_size}."
        )


def evaluate_condition(
    base_model,
    *,
    task: str,
    sequence_length: int,
    num_examples: int,
    eval_batch_size: int,
    test_seed: int,
    repetitions: int | None = None,
):
    model = make_test_model(
        task,
        base_model,
        sequence_length=sequence_length,
        batch_size=eval_batch_size,
    )

    num_batches = num_examples // eval_batch_size
    seed = fixed_condition_seed(
        test_seed,
        task,
        sequence_length,
        repetitions,
    )

    if task == "repeat-copy":
        loader = model.make_dataloader(
            num_batches=num_batches,
            seed=seed,
            fixed_repetitions=int(repetitions),
        )
    else:
        loader = model.make_dataloader(num_batches=num_batches, seed=seed)

    device = next(model.net.parameters()).device
    totals = defaultdict(float)
    seen = 0

    for _, x, y, metadata in loader:
        x = x.to(device)
        y = y.to(device)
        advice_program = model.build_advice(metadata)
        loss_mask = metadata.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        result = evaluate(
            model.net,
            model.criterion,
            x,
            y,
            advice_program=advice_program,
            loss_mask=loss_mask,
            collect_states=False,
            return_outputs=False,
        )

        b = int(y.size(1))
        seen += b
        totals["loss"] += float(result["loss"]) * b
        totals["bit_error_rate"] += float(result["bit_error_rate"]) * b
        totals["exact_sequence_accuracy"] += (
            float(result["exact_sequence_accuracy"]) * b
        )
        totals["cost"] += float(result["cost"]) * b

    if seen != num_examples:
        raise RuntimeError(f"Expected {num_examples} examples, evaluated {seen}.")

    metrics = {
        "num_examples": seen,
        "loss": totals["loss"] / seen,
        "bit_error_rate": totals["bit_error_rate"] / seen,
        "exact_sequence_accuracy": totals["exact_sequence_accuracy"] / seen,
        "cost": totals["cost"] / seen,
    }
    if task == "even-palindrome":
        metrics["classification_accuracy"] = metrics["exact_sequence_accuracy"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows, condition_fields, metric_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in condition_fields)
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        aggregate = dict(zip(condition_fields, key))
        seeds = sorted(int(row["seed"]) for row in group)
        aggregate["n_seeds"] = len(seeds)
        aggregate["seeds"] = ",".join(str(seed) for seed in seeds)

        for metric in metric_fields:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values, ddof=0))
        out.append(aggregate)
    return out


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_length_curve(aggregate, metric: str, ylabel: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for advice in ADVICES:
        rows = sorted(
            [row for row in aggregate if row["advice_type"] == advice],
            key=lambda row: int(row["sequence_length"]),
        )
        x = np.asarray([int(row["sequence_length"]) for row in rows])
        mean = np.asarray([float(row[f"mean_{metric}"]) for row in rows])
        std = np.asarray([float(row[f"std_{metric}"]) for row in rows])

        line, = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=1.8,
            label="None" if advice == "none" else "Combined",
        )
        lower = mean - std
        upper = mean + std
        if metric in {
            "bit_error_rate",
            "classification_accuracy",
            "exact_sequence_accuracy",
        }:
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    # n=20 is the upper training-length boundary.
    ax.axvline(20, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row["sequence_length"]) for row in aggregate}))
    if metric in {
        "bit_error_rate",
        "classification_accuracy",
        "exact_sequence_accuracy",
    }:
        ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.55)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.26)
    save_figure(fig, out_base)


def repeat_matrix(aggregate, advice, lengths, repetitions):
    lookup = {
        (int(row["sequence_length"]), int(row["repetitions"])):
            float(row["mean_bit_error_rate"])
        for row in aggregate
        if row["advice_type"] == advice
    }
    matrix = np.full((len(repetitions), len(lengths)), np.nan)
    for i, reps in enumerate(repetitions):
        for j, length in enumerate(lengths):
            matrix[i, j] = lookup[(int(length), int(reps))]
    return matrix


def plot_repeat_heatmap(matrix, lengths, repetitions, vmax, out_base):
    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Repetitions")
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([str(v) for v in lengths])
    ax.set_yticks(np.arange(len(repetitions)))
    ax.set_yticklabels([str(v) for v in repetitions])

    # Dashed rectangle = part of the tested grid inside the training range.
    id_cols = sum(int(n) <= 20 for n in lengths)
    id_rows = sum(int(r) <= 5 for r in repetitions)
    if id_cols and id_rows:
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                id_cols,
                id_rows,
                fill=False,
                linewidth=1.5,
                linestyle="--",
            )
        )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean bit error rate")
    fig.subplots_adjust(left=0.13, right=0.94, top=0.97, bottom=0.13)
    save_figure(fig, out_base)


def check_checkpoint_metadata(metadata, advice, seed, batch, path):
    if metadata["seed"] != int(seed):
        raise RuntimeError(f"Seed mismatch in {path}")
    if metadata["advice_type"] != advice:
        raise RuntimeError(
            f"Advice mismatch in {path}: {metadata['advice_type']} != {advice}"
        )
    if metadata["batch_num"] != int(batch):
        raise RuntimeError(f"Batch mismatch in {path}")


def evaluate_copy(args, runs_dir, out_dir, device):
    rows = []
    task_dir = runs_dir / "copy"
    seeds = TEST_SEEDS["copy"]

    print("\n=== COPY ===")
    for advice in ADVICES:
        for seed in seeds:
            path = find_checkpoint(
                task_dir / advice / f"seed_{seed}",
                args.copy_checkpoint,
            )
            model, metadata = load_base_model("copy", path, device)
            check_checkpoint_metadata(
                metadata, advice, seed, args.copy_checkpoint, path
            )
            print(f"{advice:8s} seed={seed}")

            for length in args.copy_lengths:
                metrics = evaluate_condition(
                    model,
                    task="copy",
                    sequence_length=length,
                    num_examples=args.copy_examples,
                    eval_batch_size=args.eval_batch_size,
                    test_seed=args.test_seed,
                )
                rows.append({
                    "task": "copy",
                    "advice_type": advice,
                    "seed": seed,
                    "checkpoint_batch": args.copy_checkpoint,
                    "sequence_length": length,
                    **metrics,
                })
                print(
                    f"  n={length:3d} BER={metrics['bit_error_rate']:.6f} "
                    f"exact={metrics['exact_sequence_accuracy']:.4f}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "copy_per_model.csv",
        rows,
        [
            "task", "advice_type", "seed", "checkpoint_batch",
            "sequence_length", "num_examples", "loss", "bit_error_rate",
            "exact_sequence_accuracy", "cost",
        ],
    )
    aggregate = aggregate_rows(
        rows,
        ["advice_type", "checkpoint_batch", "sequence_length"],
        ["loss", "bit_error_rate", "exact_sequence_accuracy", "cost"],
    )
    write_csv(
        out_dir / "copy_aggregate.csv",
        aggregate,
        [
            "advice_type", "checkpoint_batch", "sequence_length",
            "n_seeds", "seeds",
            "mean_loss", "std_loss",
            "mean_bit_error_rate", "std_bit_error_rate",
            "mean_exact_sequence_accuracy", "std_exact_sequence_accuracy",
            "mean_cost", "std_cost",
        ],
    )
    plot_length_curve(
        aggregate,
        "bit_error_rate",
        "Mean bit error rate",
        out_dir / "copy_generalization_ber",
    )
    return {
        "checkpoint_batch": args.copy_checkpoint,
        "seeds": list(seeds),
        "lengths": list(args.copy_lengths),
        "examples_per_length": args.copy_examples,
        "primary_metric": "bit_error_rate",
    }


def evaluate_palindrome(args, runs_dir, out_dir, device):
    if any(n <= 0 or n % 2 for n in args.palindrome_lengths):
        raise ValueError("All palindrome test lengths must be positive and even.")

    rows = []
    task_dir = runs_dir / "even-palindrome"
    seeds = TEST_SEEDS["even-palindrome"]

    print("\n=== EVEN-PALINDROME ===")
    for advice in ADVICES:
        for seed in seeds:
            path = find_checkpoint(
                task_dir / advice / f"seed_{seed}",
                args.palindrome_checkpoint,
            )
            model, metadata = load_base_model("even-palindrome", path, device)
            check_checkpoint_metadata(
                metadata, advice, seed, args.palindrome_checkpoint, path
            )
            print(f"{advice:8s} seed={seed}")

            for length in args.palindrome_lengths:
                metrics = evaluate_condition(
                    model,
                    task="even-palindrome",
                    sequence_length=length,
                    num_examples=args.palindrome_examples,
                    eval_batch_size=args.eval_batch_size,
                    test_seed=args.test_seed,
                )
                rows.append({
                    "task": "even-palindrome",
                    "advice_type": advice,
                    "seed": seed,
                    "checkpoint_batch": args.palindrome_checkpoint,
                    "sequence_length": length,
                    **metrics,
                })
                print(
                    f"  n={length:3d} "
                    f"accuracy={metrics['classification_accuracy']:.4f}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "even-palindrome_per_model.csv",
        rows,
        [
            "task", "advice_type", "seed", "checkpoint_batch",
            "sequence_length", "num_examples", "loss", "bit_error_rate",
            "exact_sequence_accuracy", "classification_accuracy", "cost",
        ],
    )
    aggregate = aggregate_rows(
        rows,
        ["advice_type", "checkpoint_batch", "sequence_length"],
        [
            "loss", "bit_error_rate", "exact_sequence_accuracy",
            "classification_accuracy", "cost",
        ],
    )
    write_csv(
        out_dir / "even-palindrome_aggregate.csv",
        aggregate,
        [
            "advice_type", "checkpoint_batch", "sequence_length",
            "n_seeds", "seeds",
            "mean_loss", "std_loss",
            "mean_bit_error_rate", "std_bit_error_rate",
            "mean_exact_sequence_accuracy", "std_exact_sequence_accuracy",
            "mean_classification_accuracy", "std_classification_accuracy",
            "mean_cost", "std_cost",
        ],
    )
    plot_length_curve(
        aggregate,
        "classification_accuracy",
        "Mean classification accuracy",
        out_dir / "even-palindrome_generalization_accuracy",
    )
    return {
        "checkpoint_batch": args.palindrome_checkpoint,
        "seeds": list(seeds),
        "lengths": list(args.palindrome_lengths),
        "examples_per_length": args.palindrome_examples,
        "primary_metric": "classification_accuracy",
    }


def evaluate_repeat(args, runs_dir, out_dir, device):
    if any(r <= 0 for r in args.repeat_repetitions):
        raise ValueError("Repeat counts must be positive.")

    rows = []
    task_dir = runs_dir / "repeat-copy"
    seeds = TEST_SEEDS["repeat-copy"]

    print("\n=== REPEAT-COPY ===")
    for advice in ADVICES:
        for seed in seeds:
            path = find_checkpoint(
                task_dir / advice / f"seed_{seed}",
                args.repeat_checkpoint,
            )
            model, metadata = load_base_model("repeat-copy", path, device)
            check_checkpoint_metadata(
                metadata, advice, seed, args.repeat_checkpoint, path
            )
            print(f"{advice:8s} seed={seed}")

            for length in args.repeat_lengths:
                for repetitions in args.repeat_repetitions:
                    metrics = evaluate_condition(
                        model,
                        task="repeat-copy",
                        sequence_length=length,
                        repetitions=repetitions,
                        num_examples=args.repeat_examples,
                        eval_batch_size=args.eval_batch_size,
                        test_seed=args.test_seed,
                    )
                    rows.append({
                        "task": "repeat-copy",
                        "advice_type": advice,
                        "seed": seed,
                        "checkpoint_batch": args.repeat_checkpoint,
                        "sequence_length": length,
                        "repetitions": repetitions,
                        **metrics,
                    })
                    print(
                        f"  n={length:3d} r={repetitions:2d} "
                        f"BER={metrics['bit_error_rate']:.6f} "
                        f"exact={metrics['exact_sequence_accuracy']:.4f}"
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "repeat-copy_per_model.csv",
        rows,
        [
            "task", "advice_type", "seed", "checkpoint_batch",
            "sequence_length", "repetitions", "num_examples", "loss",
            "bit_error_rate", "exact_sequence_accuracy", "cost",
        ],
    )
    aggregate = aggregate_rows(
        rows,
        [
            "advice_type", "checkpoint_batch", "sequence_length", "repetitions"
        ],
        ["loss", "bit_error_rate", "exact_sequence_accuracy", "cost"],
    )
    write_csv(
        out_dir / "repeat-copy_aggregate.csv",
        aggregate,
        [
            "advice_type", "checkpoint_batch", "sequence_length", "repetitions",
            "n_seeds", "seeds",
            "mean_loss", "std_loss",
            "mean_bit_error_rate", "std_bit_error_rate",
            "mean_exact_sequence_accuracy", "std_exact_sequence_accuracy",
            "mean_cost", "std_cost",
        ],
    )

    lengths = tuple(args.repeat_lengths)
    repetitions = tuple(args.repeat_repetitions)
    matrices = {
        advice: repeat_matrix(aggregate, advice, lengths, repetitions)
        for advice in ADVICES
    }
    finite = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix in matrices.values()
    ])
    vmax = max(float(np.max(finite)), 1e-12) if finite.size else 1.0

    for advice in ADVICES:
        plot_repeat_heatmap(
            matrices[advice],
            lengths,
            repetitions,
            vmax,
            out_dir / f"repeat-copy_{advice}_ber_heatmap",
        )

    difference = matrices["combined"] - matrices["none"]
    difference_rows = []
    for i, repetitions_value in enumerate(repetitions):
        for j, length in enumerate(lengths):
            difference_rows.append({
                "sequence_length": int(length),
                "repetitions": int(repetitions_value),
                "combined_minus_none_ber": float(difference[i, j]),
            })
    write_csv(
        out_dir / "repeat-copy_difference_combined_minus_none.csv",
        difference_rows,
        ["sequence_length", "repetitions", "combined_minus_none_ber"],
    )

    return {
        "checkpoint_batch": args.repeat_checkpoint,
        "seeds": list(seeds),
        "lengths": list(lengths),
        "repetitions": list(repetitions),
        "examples_per_grid_point": args.repeat_examples,
        "primary_metric": "bit_error_rate",
        "shared_heatmap_vmin": 0.0,
        "shared_heatmap_vmax": vmax,
    }


def main():
    args = parse_args()
    tasks = args.tasks or list(TASK_CLASSES.keys())
    device = resolve_device(args.device)
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_example_count(
        "--copy-examples", args.copy_examples, args.eval_batch_size
    )
    validate_example_count(
        "--palindrome-examples", args.palindrome_examples, args.eval_batch_size
    )
    validate_example_count(
        "--repeat-examples", args.repeat_examples, args.eval_batch_size
    )

    print("=" * 80)
    print("CHAPTER 5.3 -- GENERALIZATION BEYOND THE TRAINING RANGE")
    print("=" * 80)
    print(f"Runs root:       {runs_dir}")
    print(f"Output:          {out_dir}")
    print(f"Device:          {device}")
    print(f"Advice:          {ADVICES}")
    print(f"Test seed:       {args.test_seed}")
    print(f"Eval batch size: {args.eval_batch_size}")

    manifest = {
        "protocol": {
            "advices": list(ADVICES),
            "test_seed": args.test_seed,
            "eval_batch_size": args.eval_batch_size,
            "test_data": (
                "Dedicated deterministic test stream; identical examples for "
                "all training seeds and advice conditions of a task condition."
            ),
            "checkpoint_policy": (
                "Fixed periodic checkpoints, not best-validation checkpoints."
            ),
            "seed_aggregation": "mean and population SD (ddof=0)",
            "training_ranges": {
                "copy": "sequence length 2..20",
                "even-palindrome": "even sequence length 2..20",
                "repeat-copy": "sequence length 2..20, repetitions 1..5",
            },
        },
        "tasks": {},
    }

    if "copy" in tasks:
        manifest["tasks"]["copy"] = evaluate_copy(
            args, runs_dir, out_dir, device
        )
    if "even-palindrome" in tasks:
        manifest["tasks"]["even-palindrome"] = evaluate_palindrome(
            args, runs_dir, out_dir, device
        )
    if "repeat-copy" in tasks:
        manifest["tasks"]["repeat-copy"] = evaluate_repeat(
            args, runs_dir, out_dir, device
        )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nDONE")
    print(f"Results:  {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
