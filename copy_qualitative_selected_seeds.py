#!/usr/bin/env python3
"""
Create one compact qualitative Copy figure for Chapter 5.4.

Layout
------
Three seed blocks stacked vertically.

Each seed block is a 2x2 matrix:
    top-left     : no advice,  5,000 batches
    top-right    : combined,   5,000 batches
    bottom-left  : no advice, 20,000 batches
    bottom-right : combined,  20,000 batches

Inside each model cell:
    - top strip:    targets and outputs for n = 10, 20, 30, 50
    - bottom strip: target and output for n = 100

No seed labels and no checkpoint labels are shown.
Each seed block is enclosed by a thin black rectangle.
A shared colorbar is placed on the left.

Usage
-----
python copy_qualitative_selected_seeds.py
python copy_qualitative_selected_seeds.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from train import evaluate
from tasks.copytask import (
    CopyTaskModelTraining,
    CopyTaskParams,
    dataloader as copy_dataloader,
)


DEFAULT_ADVICES = ("none", "combined")
DEFAULT_SEEDS = (1001, 1002, 1004)
DEFAULT_BATCHES = (5000, 20000)
DEFAULT_LENGTHS = (10, 20, 30, 50, 100)
DEFAULT_EXAMPLE_SEED = 20260814

CHECKPOINT_RE = re.compile(r"checkpoint-batch-(\d+)\.pt$")
CMAP = "jet"


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "runs" / "copy",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis" / "copy_qualitative_selected_seeds",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCHES),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS),
    )
    parser.add_argument(
        "--example-seed",
        type=int,
        default=DEFAULT_EXAMPLE_SEED,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    return torch.device(name)


def expected_checkpoint(seed_dir: Path, batch_num: int) -> Path:
    return seed_dir / f"checkpoint-batch-{int(batch_num):08d}.pt"


def find_checkpoint(seed_dir: Path, batch_num: int) -> Path:
    canonical = expected_checkpoint(seed_dir, batch_num)

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
            f"Multiple checkpoints for batch {batch_num} in {seed_dir}: "
            + ", ".join(path.name for path in matches)
        )

    raise FileNotFoundError(
        f"Missing checkpoint for batch {batch_num}. Expected {canonical}"
    )


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_copy_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)

    if checkpoint.get("task") != "copy":
        raise ValueError(
            f"{checkpoint_path} is not a Copy checkpoint "
            f"(task={checkpoint.get('task')!r})."
        )

    params = CopyTaskParams(**checkpoint["model_params"])
    model = CopyTaskModelTraining(params=params)

    model.net.load_state_dict(checkpoint["model_state_dict"])
    model.net.to(device)
    model.net.eval()

    metadata = {
        "seed": int(checkpoint.get("seed", -1)),
        "batch_num": int(checkpoint["batch_num"]),
        "advice_type": str(params.advice_type),
    }

    return model, metadata


def make_fixed_example(
    sequence_length: int,
    sequence_width: int,
    example_seed: int,
):
    py_rng = random.Random(int(example_seed) + 10007 * int(sequence_length))
    np_rng = np.random.default_rng(int(example_seed) + 20011 * int(sequence_length))

    loader = copy_dataloader(
        num_batches=1,
        batch_size=1,
        seq_width=sequence_width,
        min_len=int(sequence_length),
        max_len=int(sequence_length),
        py_rng=py_rng,
        np_rng=np_rng,
    )

    _, x, y, metadata = next(iter(loader))
    return x, y, metadata


def evaluate_model_on_example(
    model,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    metadata: dict,
):
    x = x.to(device)
    y = y.to(device)

    advice_program = model.build_advice(metadata)

    result = evaluate(
        model.net,
        model.criterion,
        x,
        y,
        advice_program=advice_program,
        collect_states=False,
        return_outputs=True,
    )

    output = result["y_out"].detach().cpu().numpy()[:, 0, :]
    target = y.detach().cpu().numpy()[:, 0, :]

    return output, target


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def draw_heatmap(ax, matrix: np.ndarray):
    image = ax.imshow(
        matrix.T,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def add_model_cell(
    fig,
    subplot_spec,
    model_results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    lengths,
):
    short_lengths = lengths[:4]
    long_length = lengths[4]

    axes_created: List[plt.Axes] = []

    # smaller gap between upper block (10,20,30,50) and lower block (100)
    cell = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[1.0, 1.05],
        hspace=0.10,
    )

    short_block = GridSpecFromSubplotSpec(
        2,
        len(short_lengths),
        subplot_spec=cell[0],
        height_ratios=[1.0, 1.0],
        width_ratios=list(short_lengths),
        hspace=0.05,
        wspace=0.025,
    )

    last_image = None

    for col, length in enumerate(short_lengths):
        output, target = model_results[length]

        ax_target = fig.add_subplot(short_block[0, col])
        draw_heatmap(ax_target, target)
        axes_created.append(ax_target)

        ax_output = fig.add_subplot(short_block[1, col])
        last_image = draw_heatmap(ax_output, output)
        axes_created.append(ax_output)

    long_block = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=cell[1],
        height_ratios=[1.0, 1.0],
        hspace=0.05,
    )

    output, target = model_results[long_length]

    ax_target = fig.add_subplot(long_block[0])
    draw_heatmap(ax_target, target)
    axes_created.append(ax_target)

    ax_output = fig.add_subplot(long_block[1])
    last_image = draw_heatmap(ax_output, output)
    axes_created.append(ax_output)

    return last_image, axes_created


def save_figure(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        base_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def add_block_rectangle(fig, axes_list, pad_x=0.008, pad_y=0.010):
    x0 = min(ax.get_position().x0 for ax in axes_list)
    y0 = min(ax.get_position().y0 for ax in axes_list)
    x1 = max(ax.get_position().x1 for ax in axes_list)
    y1 = max(ax.get_position().y1 for ax in axes_list)

    rect = Rectangle(
        (x0 - pad_x, y0 - pad_y),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) + 2 * pad_y,
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        transform=fig.transFigure,
        zorder=20,
    )
    fig.add_artist(rect)


def plot_selected_matrix(
    results_by_model,
    *,
    advices,
    seeds,
    batches,
    lengths,
    out_base: Path,
):
    if tuple(advices) != ("none", "combined"):
        raise ValueError("This layout requires exactly: none combined")

    if len(lengths) != 5:
        raise ValueError("This layout requires exactly five sequence lengths.")

    if len(seeds) != 3:
        raise ValueError("This layout requires exactly three selected seeds.")

    if len(batches) != 2:
        raise ValueError("This layout requires exactly two checkpoints.")

    fig = plt.figure(figsize=(12.8, 14.5))

    outer = GridSpec(
        3,
        1,
        figure=fig,
        left=0.10,
        right=0.93,
        bottom=0.03,
        top=0.94,
        hspace=0.16,
    )

    # keep only the advice headings
    fig.text(0.325, 0.955, "No advice", ha="center", va="center", fontsize=11)
    fig.text(0.705, 0.955, "Combined advice", ha="center", va="center", fontsize=11)

    last_image = None

    # seed blocks
    for seed_idx, seed in enumerate(seeds):
        seed_spec = outer[seed_idx]

        inner = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=seed_spec,
            hspace=0.12,
            wspace=0.10,
        )

        seed_axes = []

        # top row = 5000, bottom row = 20000
        layout = [
            (0, 0, batches[0], "none"),
            (0, 1, batches[0], "combined"),
            (1, 0, batches[1], "none"),
            (1, 1, batches[1], "combined"),
        ]

        for r, c, batch_num, advice in layout:
            last_image, axes_created = add_model_cell(
                fig,
                inner[r, c],
                results_by_model[(batch_num, advice, seed)],
                lengths,
            )
            seed_axes.extend(axes_created)

        add_block_rectangle(fig, seed_axes)

    if last_image is not None:
        colorbar_ax = fig.add_axes([0.035, 0.13, 0.018, 0.74])

        colorbar = fig.colorbar(last_image, cax=colorbar_ax)
        colorbar.set_ticks(np.linspace(0.0, 1.0, 11))
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.set_label("Output probability", fontsize=9)

    save_figure(fig, out_base)


def main():
    args = parse_args()
    configure_plot_style()

    advices = DEFAULT_ADVICES
    seeds = tuple(args.seeds)
    batches = tuple(args.batches)
    lengths = tuple(args.lengths)

    if len(seeds) != 3:
        raise ValueError(
            "The main-text figure is designed for exactly three seeds."
        )

    if len(batches) != 2:
        raise ValueError(
            "The main-text figure is designed for exactly two checkpoints."
        )

    if len(lengths) != 5:
        raise ValueError("--lengths must contain exactly five lengths.")

    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    device = resolve_device(args.device)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("COPY QUALITATIVE ANALYSIS -- SELECTED SEEDS")
    print("=" * 78)
    print(f"Runs:         {runs_dir}")
    print(f"Output:       {out_dir}")
    print(f"Device:       {device}")
    print(f"Seeds:        {seeds}")
    print(f"Checkpoints:  {batches}")
    print(f"Lengths:      {lengths}")
    print(f"Example seed: {args.example_seed}")
    print()

    first_checkpoint = find_checkpoint(
        runs_dir / "none" / f"seed_{seeds[0]}",
        batches[0],
    )

    first_model, _ = load_copy_model(first_checkpoint, device)
    sequence_width = int(first_model.params.sequence_width)
    del first_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    fixed_examples = {
        length: make_fixed_example(
            length,
            sequence_width,
            args.example_seed,
        )
        for length in lengths
    }

    results_by_model = {}

    for batch_num in batches:
        print(f"\n--- batch {batch_num:,} ---")

        for advice in advices:
            advice_dir = runs_dir / advice

            if not advice_dir.exists():
                raise FileNotFoundError(f"Missing advice directory: {advice_dir}")

            for seed in seeds:
                seed_dir = advice_dir / f"seed_{seed}"
                checkpoint_path = find_checkpoint(seed_dir, batch_num)

                print(f"{advice:8s} | seed={seed} | {checkpoint_path.name}")

                model, metadata = load_copy_model(checkpoint_path, device)

                if metadata["advice_type"] != advice:
                    raise RuntimeError(f"Advice mismatch for {checkpoint_path}")
                if metadata["seed"] != seed:
                    raise RuntimeError(f"Seed mismatch for {checkpoint_path}")
                if metadata["batch_num"] != batch_num:
                    raise RuntimeError(f"Batch mismatch for {checkpoint_path}")

                model_results = {}

                for length in lengths:
                    x, y, task_metadata = fixed_examples[length]

                    output, target = evaluate_model_on_example(
                        model,
                        device,
                        x,
                        y,
                        task_metadata,
                    )

                    model_results[length] = (output, target)

                results_by_model[(batch_num, advice, seed)] = model_results

                del model

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    out_base = out_dir / "copy_qualitative_selected_seeds"

    plot_selected_matrix(
        results_by_model,
        advices=advices,
        seeds=seeds,
        batches=batches,
        lengths=lengths,
        out_base=out_base,
    )

    manifest = {
        "selected_seeds": list(seeds),
        "selected_checkpoints": list(batches),
        "sequence_lengths": list(lengths),
        "example_seed": int(args.example_seed),
        "advice_conditions": list(advices),
    }

    manifest_path = out_dir / "selection_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print()
    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {out_base}.png")
    print(f"Saved: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()