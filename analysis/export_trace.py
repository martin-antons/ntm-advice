#!/usr/bin/env python
"""Export one deterministic NTM evaluation trace to a compressed NPZ file.

The script is intentionally separate from training. It loads a checkpoint,
generates one fixed task batch, optionally changes the advice condition and
lambda value, and stores arrays required for read/write/advice/memory heatmaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import attr
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import evaluate, resolve_device  # noqa: E402
from tasks.copytask import CopyTaskModelTraining, CopyTaskParams  # noqa: E402
from tasks.repeatcopytask import (  # noqa: E402
    RepeatCopyTaskModelTraining,
    RepeatCopyTaskParams,
)
from tasks.evenpalindrometask import (  # noqa: E402
    EvenPalindromeTaskModelTraining,
    EvenPalindromeTaskParams,
)


TASKS = {
    "copy": (CopyTaskModelTraining, CopyTaskParams),
    "repeat-copy": (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    "even-palindrome": (
        EvenPalindromeTaskModelTraining,
        EvenPalindromeTaskParams,
    ),
}


def _tensor_stack(trace, key):
    if not trace:
        return np.empty((0,), dtype=np.float32)
    return torch.stack([entry[key] for entry in trace], dim=0).cpu().numpy()


def _jsonable_metadata(metadata):
    result = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _build_params(task, params_cls, model_params, args):
    valid = {field.name for field in attr.fields(params_cls)}
    filtered = {
        key: value
        for key, value in model_params.items()
        if key in valid
    }
    params = params_cls(**filtered)

    updates = {
        "num_batches": 1,
        "batch_size": args.batch_size,
        "sequence_min_len": args.length,
        "sequence_max_len": args.length,
    }
    if args.advice_type is not None:
        updates["advice_type"] = args.advice_type
    return attr.evolve(params, **updates)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=20260723)
    parser.add_argument("--advice-type", default=None)
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    task = args.task or checkpoint.get("task")
    if task not in TASKS:
        raise ValueError(
            "Task is missing or unsupported; pass --task explicitly"
        )

    model_cls, params_cls = TASKS[task]
    params = _build_params(
        task,
        params_cls,
        checkpoint["model_params"],
        args,
    )
    model = model_cls(params=params)
    model.net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.net.to(device)

    if args.lambda_value is not None:
        model.net.set_advice_strength(args.lambda_value)

    dataloader_kwargs = {
        "num_batches": 1,
        "seed": args.data_seed,
    }

    if task == "repeat-copy":
        dataloader_kwargs["fixed_repetitions"] = (
            args.repetitions
        )

    batch = next(
        iter(model.make_dataloader(**dataloader_kwargs))
    )
    _, x, y, metadata = batch
    x = x.to(device)
    y = y.to(device)
    advice_program = model.build_advice(metadata)

    result = evaluate(
        model.net,
        model.criterion,
        x,
        y,
        advice_program=advice_program,
        loss_mask=metadata.get("loss_mask"),
        record_advice_trace=True,
        record_ntm_trace=True,
        collect_states=False,
        return_outputs=True,
    )

    ntm_trace = result["ntm_trace"]
    advice_trace = result["advice_trace"]

    if advice_trace and advice_trace[0]["vector"] is not None:
        advice_vectors = torch.stack(
            [entry["vector"] for entry in advice_trace],
            dim=0,
        ).cpu().numpy()
        advice_indices = np.stack([
            (
                entry["indices"].cpu().numpy()
                if entry["indices"] is not None
                else np.full((args.batch_size,), -1, dtype=np.int64)
            )
            for entry in advice_trace
        ])
        advice_weightings = np.stack([
            (
                entry["weighting"].cpu().numpy()
                if entry["weighting"] is not None
                else np.empty((args.batch_size, 0), dtype=np.float32)
            )
            for entry in advice_trace
        ])
    else:
        advice_vectors = np.empty(
            (len(ntm_trace), args.batch_size, 0),
            dtype=np.float32,
        )
        advice_indices = np.full(
            (len(ntm_trace), args.batch_size),
            -1,
            dtype=np.int64,
        )
        advice_weightings = np.empty(
            (len(ntm_trace), args.batch_size, 0),
            dtype=np.float32,
        )

    payload = {
        "inputs": x.detach().cpu().numpy(),
        "targets": y.detach().cpu().numpy(),
        "predictions": result["y_out"].detach().cpu().numpy(),
        "predictions_binary": result[
            "y_out_binarized"
        ].detach().cpu().numpy(),
        "controller_outputs": _tensor_stack(
            ntm_trace,
            "controller_output",
        ),
        "read_vectors": _tensor_stack(ntm_trace, "read_vectors"),
        "read_weightings": _tensor_stack(
            ntm_trace,
            "read_weightings",
        ),
        "write_weightings": _tensor_stack(
            ntm_trace,
            "write_weightings",
        ),
        "memory_before": _tensor_stack(ntm_trace, "memory_before"),
        "memory_after": _tensor_stack(ntm_trace, "memory_after"),
        "model_outputs_all_steps": _tensor_stack(ntm_trace, "output"),
        "advice_vectors": advice_vectors,
        "advice_indices": advice_indices,
        "advice_weightings": advice_weightings,
        "metadata_json": np.asarray(
            json.dumps(_jsonable_metadata(metadata))
        ),
        "evaluation_json": np.asarray(
            json.dumps({
                "task": task,
                "checkpoint": str(args.checkpoint),
                "checkpoint_batch": checkpoint.get("batch_num"),
                "trained_advice_type": checkpoint.get(
                    "model_params",
                    {},
                ).get("advice_type"),
                "evaluated_advice_type": params.advice_type,
                "lambda_value": model.net.advice_strength,
                "length": args.length,
                "repetitions": (
                    args.repetitions if task == "repeat-copy" else None
                ),
                "loss": result["loss"],
                "cost": result["cost"],
                "bit_error_rate": result["bit_error_rate"],
                "exact_sequence_accuracy": result[
                    "exact_sequence_accuracy"
                ],
            })
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        json.dumps({
            "output": str(args.output),
            "task": task,
            "loss": result["loss"],
            "bit_error_rate": result["bit_error_rate"],
            "exact_sequence_accuracy": result[
                "exact_sequence_accuracy"
            ],
            "model_steps": len(ntm_trace),
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
