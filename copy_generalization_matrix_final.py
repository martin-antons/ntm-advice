#!/usr/bin/env python3
"""
Create compact Figure-4-style Copy generalization matrices.

For each checkpoint batch, one large figure is produced:

    columns = No advice | Combined advice
    rows    = seeds 1000 ... 1005

Inside every seed/advice cell the layout mirrors the visual structure of
Figure 4 in Graves et al.:

    top block:
        targets for n = 10, 20, 30, 50
        outputs for n = 10, 20, 30, 50

    bottom block:
        target for n = 100
        output for n = 100

Important visual choices:
- Targets are ABOVE outputs.
- No labels for lengths, targets, or outputs inside the cells.
- The four short sequences are placed side-by-side with widths proportional
  to their actual sequence lengths.
- There is only a small gap between target and output, and a larger gap
  between the short-sequence block and n=100.
- One shared color scale is shown at the far right:
      0.0 -> dark blue
      1.0 -> dark red
      intermediate values -> bright multi-colour values
- Only seed labels and advice-column labels remain.

Expected repository structure
-----------------------------
Place this file next to train.py:

<repo>/
    train.py
    copy_generalization_matrix_final.py
    tasks/
    ntm/
    runs/
        copy/
            none/
                seed_1000/
                    checkpoint-batch-00005000.pt
                    checkpoint-batch-00010000.pt
                    checkpoint-batch-00015000.pt
                    checkpoint-batch-00020000.pt
                ...
                seed_1005/
            combined/
                seed_1000/
                ...
                seed_1005/

Usage
-----
python copy_generalization_matrix_final.py
python copy_generalization_matrix_final.py --device cuda
python copy_generalization_matrix_final.py --device cuda --batches 5000
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from train import evaluate
from tasks.copytask import (
    CopyTaskModelTraining,
    CopyTaskParams,
    dataloader as copy_dataloader,
)


DEFAULT_ADVICES = ("none", "combined")
DEFAULT_SEEDS = (1000, 1001, 1002, 1003, 1004, 1005)
DEFAULT_BATCHES = (5000, 10000, 15000, 20000)
DEFAULT_LENGTHS = (10, 20, 30, 50, 100)

CHECKPOINT_RE = re.compile(r"checkpoint-batch-(\d+)\.pt$")

# Matches the qualitative colour scale in the supplied Figure-4 example:
# low values blue, high values red, bright colours in between.
CMAP = "jet"


def parse_args():
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Create compact Figure-4-style Copy generalization matrices "
            "with targets above outputs."
        )
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "runs" / "copy",
        help="Default: <script-dir>/runs/copy",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "analysis" / "copy_generalization_matrix_final",
        help="Output directory.",
    )
    parser.add_argument(
        "--advices",
        nargs="+",
        default=list(DEFAULT_ADVICES),
        help="Must be exactly: none combined",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds. Default: 1000 ... 1005",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCHES),
        help="Checkpoint batches. Default: 5000 10000 15000 20000",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS),
        help="Exactly five lengths. Default: 10 20 30 50 100",
    )
    parser.add_argument(
        "--example-seed",
        type=int,
        default=20260814,
        help=(
            "Seed for the fixed qualitative examples. Every model receives "
            "the same concrete example for a given sequence length."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )

    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    return torch.device(name)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def expected_checkpoint(seed_dir: Path, batch_num: int) -> Path:
    return seed_dir / f"checkpoint-batch-{batch_num:08d}.pt"


def find_checkpoint(seed_dir: Path, batch_num: int) -> Path:
    canonical = expected_checkpoint(seed_dir, batch_num)

    if canonical.exists():
        return canonical

    matches = []

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
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_copy_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load_checkpoint(
        checkpoint_path,
        device,
    )

    if checkpoint.get("task") != "copy":
        raise ValueError(
            f"{checkpoint_path} is not a Copy checkpoint "
            f"(task={checkpoint.get('task')!r})."
        )

    params = CopyTaskParams(
        **checkpoint["model_params"]
    )

    model = CopyTaskModelTraining(
        params=params
    )

    model.net.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.net.to(device)
    model.net.eval()

    metadata = {
        "seed": int(checkpoint.get("seed", -1)),
        "batch_num": int(checkpoint["batch_num"]),
        "advice_type": str(params.advice_type),
    }

    return model, metadata


# ---------------------------------------------------------------------------
# Fixed qualitative examples
# ---------------------------------------------------------------------------

def make_fixed_example(
    sequence_length: int,
    sequence_width: int,
    example_seed: int,
):
    """
    Generate one deterministic Copy example for a fixed sequence length.

    The generators depend only on example_seed and sequence_length, so all
    models see exactly the same qualitative example.
    """
    py_rng = random.Random(
        example_seed + 10007 * sequence_length
    )

    np_rng = np.random.default_rng(
        example_seed + 20011 * sequence_length
    )

    loader = copy_dataloader(
        num_batches=1,
        batch_size=1,
        seq_width=sequence_width,
        min_len=sequence_length,
        max_len=sequence_length,
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

    # (time, batch=1, bits) -> (time, bits)
    output = (
        result["y_out"]
        .detach()
        .cpu()
        .numpy()[:, 0, :]
    )

    target = (
        y.detach()
        .cpu()
        .numpy()[:, 0, :]
    )

    return output, target


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def draw_heatmap(ax, matrix: np.ndarray):
    """
    Plot time horizontally and Copy bits vertically.
    matrix shape = (time, bits)
    """
    image = ax.imshow(
        matrix.T,
        origin="upper",
        aspect="auto",
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


def save_figure(fig, base_path: Path):
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
    )

    fig.savefig(
        base_path.with_suffix(".png"),
        bbox_inches="tight",
        pad_inches=0.025,
    )

    plt.close(fig)


def add_model_cell(
    fig,
    subplot_spec,
    model_results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    lengths,
):
    """
    Build one Figure-4-like cell.

    Top block:
        targets n=10,20,30,50
        outputs n=10,20,30,50

    Bottom block:
        target n=100
        output n=100

    Widths of the four top examples are proportional to their actual lengths.
    """
    short_lengths = lengths[:4]
    long_length = lengths[4]

    # Larger gap between the short block and the n=100 block.
    cell = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[1.0, 1.18],
        hspace=0.22,
    )

    # -------------------------
    # Top: n=10,20,30,50
    # -------------------------
    short_block = GridSpecFromSubplotSpec(
        2,
        len(short_lengths),
        subplot_spec=cell[0],
        height_ratios=[1.0, 1.0],
        width_ratios=list(short_lengths),
        hspace=0.10,
        wspace=0.055,
    )

    last_image = None

    # Targets ABOVE outputs.
    for col, length in enumerate(short_lengths):
        output, target = model_results[length]

        ax_target = fig.add_subplot(
            short_block[0, col]
        )
        draw_heatmap(
            ax_target,
            target,
        )

        ax_output = fig.add_subplot(
            short_block[1, col]
        )
        last_image = draw_heatmap(
            ax_output,
            output,
        )

    # -------------------------
    # Bottom: n=100
    # -------------------------
    long_block = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=cell[1],
        height_ratios=[1.0, 1.0],
        hspace=0.10,
    )

    output, target = model_results[
        long_length
    ]

    ax_target = fig.add_subplot(
        long_block[0]
    )
    draw_heatmap(
        ax_target,
        target,
    )

    ax_output = fig.add_subplot(
        long_block[1]
    )
    last_image = draw_heatmap(
        ax_output,
        output,
    )

    return last_image


def plot_batch_matrix(
    results_by_model: Dict[
        Tuple[str, int],
        Dict[int, Tuple[np.ndarray, np.ndarray]],
    ],
    *,
    advices,
    seeds,
    lengths,
    batch_num: int,
    out_base: Path,
):
    if tuple(advices) != ("none", "combined"):
        raise ValueError(
            "This layout requires exactly: none combined"
        )

    if len(lengths) != 5:
        raise ValueError(
            "This layout requires exactly five sequence lengths."
        )

    # Tall enough that each of the 12 model cells remains readable.
    fig = plt.figure(
        figsize=(13.6, 18.4)
    )

    # Main 6 x 2 matrix.
    outer = GridSpec(
        len(seeds),
        len(advices),
        figure=fig,
        left=0.075,
        right=0.905,
        bottom=0.025,
        top=0.945,
        hspace=0.14,
        wspace=0.075,
    )

    # Batch title.
    fig.text(
        0.49,
        0.988,
        f"{batch_num:,} training batches",
        ha="center",
        va="top",
        fontsize=11,
    )

    # Advice labels.
    fig.text(
        0.285,
        0.961,
        "No advice",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.text(
        0.695,
        0.961,
        "Combined advice",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Seed labels on left.
    usable_top = 0.945
    usable_bottom = 0.025
    usable_height = usable_top - usable_bottom

    for row, seed in enumerate(seeds):
        y = (
            usable_top
            - (row + 0.5)
            * usable_height
            / len(seeds)
        )

        fig.text(
            0.035,
            y,
            str(seed),
            ha="center",
            va="center",
            fontsize=9,
        )

    last_image = None

    for row, seed in enumerate(seeds):
        for col, advice in enumerate(advices):
            last_image = add_model_cell(
                fig,
                outer[row, col],
                results_by_model[
                    (advice, seed)
                ],
                lengths,
            )

    # One large shared colour bar.
    if last_image is not None:
        colorbar_ax = fig.add_axes(
            [0.925, 0.10, 0.020, 0.79]
        )

        colorbar = fig.colorbar(
            last_image,
            cax=colorbar_ax,
        )

        colorbar.set_ticks(
            np.linspace(0.0, 1.0, 11)
        )

        colorbar.ax.tick_params(
            labelsize=8,
            length=2,
        )

    save_figure(
        fig,
        out_base,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    configure_plot_style()

    advices = tuple(
        str(advice)
        .strip()
        .lower()
        .replace("-", "_")
        for advice in args.advices
    )

    seeds = tuple(args.seeds)
    batches = tuple(args.batches)
    lengths = tuple(args.lengths)

    if advices != ("none", "combined"):
        raise ValueError(
            "Please use exactly: --advices none combined"
        )

    if len(lengths) != 5:
        raise ValueError(
            "--lengths must contain exactly five lengths."
        )

    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    device = resolve_device(
        args.device
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("COPY GENERALIZATION MATRIX — FINAL FIGURE-4 LAYOUT")
    print("=" * 78)
    print(f"Runs:         {runs_dir}")
    print(f"Output:       {out_dir}")
    print(f"Device:       {device}")
    print(f"Advices:      {advices}")
    print(f"Seeds:        {seeds}")
    print(f"Checkpoints:  {batches}")
    print(f"Lengths:      {lengths}")
    print(f"Example seed: {args.example_seed}")
    print()

    # Determine Copy vector width from the first checkpoint.
    first_checkpoint = find_checkpoint(
        runs_dir
        / "none"
        / f"seed_{seeds[0]}",
        batches[0],
    )

    first_model, _ = load_copy_model(
        first_checkpoint,
        device,
    )

    sequence_width = int(
        first_model.params.sequence_width
    )

    del first_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Same five concrete test examples for every model.
    fixed_examples = {
        length: make_fixed_example(
            length,
            sequence_width,
            args.example_seed,
        )
        for length in lengths
    }

    for batch_num in batches:

        print(
            f"\n--- batch {batch_num} ---"
        )

        results_by_model = {}

        for advice in advices:

            advice_dir = (
                runs_dir / advice
            )

            if not advice_dir.exists():
                raise FileNotFoundError(
                    f"Missing advice directory: "
                    f"{advice_dir}"
                )

            for seed in seeds:

                seed_dir = (
                    advice_dir
                    / f"seed_{seed}"
                )

                checkpoint_path = find_checkpoint(
                    seed_dir,
                    batch_num,
                )

                print(
                    f"{advice:8s} | "
                    f"seed={seed} | "
                    f"{checkpoint_path.name}"
                )

                model, metadata = load_copy_model(
                    checkpoint_path,
                    device,
                )

                if metadata["advice_type"] != advice:
                    raise RuntimeError(
                        "Advice mismatch: "
                        f"directory={advice}, "
                        f"checkpoint="
                        f"{metadata['advice_type']}"
                    )

                if metadata["seed"] != seed:
                    raise RuntimeError(
                        "Seed mismatch: "
                        f"directory={seed}, "
                        f"checkpoint="
                        f"{metadata['seed']}"
                    )

                if metadata["batch_num"] != batch_num:
                    raise RuntimeError(
                        "Batch mismatch: "
                        f"requested={batch_num}, "
                        f"checkpoint="
                        f"{metadata['batch_num']}"
                    )

                model_results = {}

                for length in lengths:

                    x, y, task_metadata = (
                        fixed_examples[length]
                    )

                    output, target = (
                        evaluate_model_on_example(
                            model,
                            device,
                            x,
                            y,
                            task_metadata,
                        )
                    )

                    model_results[length] = (
                        output,
                        target,
                    )

                results_by_model[
                    (advice, seed)
                ] = model_results

                del model

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        out_base = (
            out_dir
            / (
                "copy_generalization_"
                f"batch_{batch_num:05d}"
            )
        )

        plot_batch_matrix(
            results_by_model,
            advices=advices,
            seeds=seeds,
            lengths=lengths,
            batch_num=batch_num,
            out_base=out_base,
        )

        print(
            f"Saved: {out_base}.pdf/.png"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
