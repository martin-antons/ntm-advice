#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training entry point for baseline and advice-augmented NTM tasks.

The script writes one canonical ``history.json`` per run. Periodic model
checkpoints are separate ``.pt`` files, so the complete metric arrays are not
copied into a new JSON file at every checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Optional

try:
    import argcomplete
except ImportError:  # optional CLI convenience
    argcomplete = None

import attr
import numpy as np
import torch
import torch.nn.functional as F

from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.evenpalindrometask import (
    EvenPalindromeTaskModelTraining,
    EvenPalindromeTaskParams,
)


LOGGER = logging.getLogger(__name__)

TASKS = {
    "copy": (CopyTaskModelTraining, CopyTaskParams),
    "repeat-copy": (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    "even-palindrome": (
        EvenPalindromeTaskModelTraining,
        EvenPalindromeTaskParams,
    ),
}

RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000
HISTORY_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_ms() -> float:
    return time.time() * 1000.0


def init_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            LOGGER.warning(
                "Strict deterministic algorithms are unavailable for this setup."
            )

    return seed


def progress_clean() -> None:
    print("\r{}".format(" " * 100), end="\r")


def progress_bar(batch_num: int, report_interval: int, last_loss: float) -> None:
    progress = (((batch_num - 1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print(
        "\r[{}{}]: {} (Loss: {:.4f})".format(
            "=" * fill,
            " " * (40 - fill),
            batch_num,
            last_loss,
        ),
        end="",
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid a truncated history after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _new_history(model, args) -> Dict[str, Any]:
    model_params = attr.asdict(model.params)
    advice_type = str(model_params.get("advice_type", "none"))
    run_id = args.run_id or "{}-{}-seed-{}".format(
        args.task,
        advice_type,
        args.seed,
    )
    now = _utc_now()

    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "task": args.task,
            "advice_type": advice_type,
            "seed": int(args.seed),
            "status": "running",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "last_batch": 0,
            "report_interval": int(args.report_interval),
            "checkpoint_interval": int(args.checkpoint_interval),
            "parameter_count": int(model.net.calculate_num_params()),
            "deterministic": bool(args.deterministic),
        },
        "model_params": model_params,
        "metrics": {
            "batch": [],
            "loss": [],
            "cost": [],
            "sequence_length": [],
        },
        "reports": {
            "batch": [],
            "mean_loss": [],
            "mean_cost": [],
            "ms_per_sequence": [],
            "advice_weight_norm": [],
            "advice_gradient_norm": [],
        },
    }


def _save_history(history: Dict[str, Any], run_dir: Path) -> None:
    history["run"]["updated_at"] = _utc_now()
    _atomic_write_json(run_dir / "history.json", history)


def save_checkpoint(
    model,
    args,
    batch_num: int,
    history: Dict[str, Any],
    *,
    label: Optional[str] = None,
) -> Path:
    """Save model/optimizer state and update the canonical history JSON."""
    progress_clean()
    run_dir = Path(args.checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    if label is None:
        filename = "checkpoint-batch-{:08d}.pt".format(batch_num)
    else:
        filename = "checkpoint-{}.pt".format(label)
    checkpoint_path = run_dir / filename

    checkpoint = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "batch_num": int(batch_num),
        "seed": int(args.seed),
        "task": args.task,
        "run": history["run"],
        "model_params": history["model_params"],
        "model_state_dict": model.net.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        # Included for convenient resume/inspection from a single file.
        "metrics": history["metrics"],
        "reports": history["reports"],
    }

    LOGGER.info("Saving checkpoint to '%s'", checkpoint_path)
    torch.save(checkpoint, checkpoint_path)
    _save_history(history, run_dir)
    return checkpoint_path


def clip_grads(net) -> None:
    for parameter in net.parameters():
        if parameter.grad is not None:
            parameter.grad.data.clamp_(-10, 10)


def _compute_loss(criterion, prediction, target, loss_mask=None):
    if loss_mask is None:
        return criterion(prediction, target)

    loss_mask = loss_mask.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if loss_mask.shape != prediction.shape:
        raise ValueError(
            "loss_mask shape {} does not match prediction shape {}".format(
                tuple(loss_mask.shape),
                tuple(prediction.shape),
            )
        )

    elementwise = F.binary_cross_entropy(
        prediction,
        target,
        reduction="none",
    )
    normalizer = loss_mask.sum().clamp_min(1.0)
    return (elementwise * loss_mask).sum() / normalizer


def _compute_cost(prediction, target, batch_size, loss_mask=None) -> float:
    binary = prediction.detach().ge(0.5).to(target.dtype)
    errors = torch.abs(binary - target.detach())
    if loss_mask is not None:
        errors = errors * loss_mask.to(
            device=errors.device,
            dtype=errors.dtype,
        )
    return errors.sum().item() / batch_size


def _init_model_sequence(net, batch_size, advice_program=None, trace=False):
    if advice_program is None:
        net.init_sequence(
            batch_size,
            record_advice_trace=trace,
        )
    else:
        net.init_sequence(
            batch_size,
            advice=advice_program.matrix,
            advice_schedule=advice_program.schedule,
            record_advice_trace=trace,
        )


def _advice_diagnostics(net) -> Optional[Dict[str, float]]:
    """Inspect the advice columns in the first LSTM input matrix.

    Advice is concatenated only to the input of LSTM layer 0. Therefore the
    final ``advice_size`` columns of ``weight_ih_l0`` are the relevant weights.
    """
    advice_size = int(getattr(net, "advice_size", 0))
    if advice_size <= 0:
        return None

    weight = net.ntm.controller.lstm.weight_ih_l0
    if advice_size > weight.size(1):
        raise RuntimeError("advice_size exceeds the controller input width")

    advice_weight = weight[:, -advice_size:]
    gradient_norm = 0.0
    if weight.grad is not None:
        gradient_norm = float(weight.grad[:, -advice_size:].norm().item())

    return {
        "weight_norm": float(advice_weight.norm().item()),
        "gradient_norm": gradient_norm,
    }


def train_batch(
    net,
    criterion,
    optimizer,
    X,
    Y,
    *,
    advice_program=None,
    loss_mask=None,
):
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    _init_model_sequence(net, batch_size, advice_program)

    for step in range(inp_seq_len):
        net(X[step])

    y_out = Y.new_zeros(Y.size())
    for step in range(outp_seq_len):
        y_out[step], _ = net()

    loss = _compute_loss(criterion, y_out, Y, loss_mask)
    loss.backward()
    clip_grads(net)

    # Gradient must be inspected before optimizer.step()/zero_grad().
    diagnostics = _advice_diagnostics(net)
    optimizer.step()

    # Weight norm after the update is more intuitive in the report.
    if diagnostics is not None:
        updated = _advice_diagnostics(net)
        diagnostics["weight_norm"] = updated["weight_norm"]

    cost = _compute_cost(y_out, Y, batch_size, loss_mask)
    return loss.item(), cost, diagnostics


def evaluate(
    net,
    criterion,
    X,
    Y,
    *,
    advice_program=None,
    loss_mask=None,
    record_advice_trace=False,
):
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    with torch.no_grad():
        _init_model_sequence(
            net,
            batch_size,
            advice_program,
            trace=record_advice_trace,
        )

        states = []
        for step in range(inp_seq_len):
            _, state = net(X[step])
            states.append(state)

        y_out = Y.new_zeros(Y.size())
        for step in range(outp_seq_len):
            y_out[step], state = net()
            states.append(state)

        loss = _compute_loss(criterion, y_out, Y, loss_mask)
        cost = _compute_cost(y_out, Y, batch_size, loss_mask)

    return {
        "loss": loss.item(),
        "cost": cost,
        "y_out": y_out,
        "y_out_binarized": y_out.ge(0.5).to(Y.dtype),
        "states": states,
        "advice_trace": list(net.advice_trace),
    }


def _unpack_batch(batch):
    if len(batch) == 3:
        batch_num, x, y = batch
        metadata = {}
    elif len(batch) == 4:
        batch_num, x, y, metadata = batch
    else:
        raise ValueError(
            "Dataloader batches must contain 3 or 4 entries, got {}".format(
                len(batch)
            )
        )
    return batch_num, x, y, metadata


def _build_advice(model, metadata):
    if not hasattr(model, "build_advice"):
        return None
    return model.build_advice(metadata)


def _append_report(
    history: Dict[str, Any],
    batch_num: int,
    mean_loss: float,
    mean_cost: float,
    ms_per_sequence: int,
    diagnostics: Optional[Dict[str, float]],
) -> None:
    reports = history["reports"]
    reports["batch"].append(int(batch_num))
    reports["mean_loss"].append(float(mean_loss))
    reports["mean_cost"].append(float(mean_cost))
    reports["ms_per_sequence"].append(int(ms_per_sequence))
    reports["advice_weight_norm"].append(
        None if diagnostics is None else float(diagnostics["weight_norm"])
    )
    reports["advice_gradient_norm"].append(
        None if diagnostics is None else float(diagnostics["gradient_norm"])
    )


def train_model(model, args):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    run_dir = Path(args.checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Training model for %d batches (batch_size=%d)...",
        num_batches,
        batch_size,
    )

    history = _new_history(model, args)
    _save_history(history, run_dir)

    start_ms = get_ms()
    batch_num = 0
    last_diagnostics = None

    try:
        for batch in model.dataloader:
            batch_num, x, y, metadata = _unpack_batch(batch)
            advice_program = _build_advice(model, metadata)
            loss_mask = metadata.get("loss_mask")

            loss, cost, diagnostics = train_batch(
                model.net,
                model.criterion,
                model.optimizer,
                x,
                y,
                advice_program=advice_program,
                loss_mask=loss_mask,
            )
            last_diagnostics = diagnostics

            metrics = history["metrics"]
            metrics["batch"].append(int(batch_num))
            metrics["loss"].append(float(loss))
            metrics["cost"].append(float(cost))
            metrics["sequence_length"].append(
                int(metadata.get("sequence_length", y.size(0)))
            )
            history["run"]["last_batch"] = int(batch_num)

            progress_bar(batch_num, args.report_interval, loss)

            if batch_num % args.report_interval == 0:
                mean_loss = float(
                    np.asarray(metrics["loss"][-args.report_interval:]).mean()
                )
                mean_cost = float(
                    np.asarray(metrics["cost"][-args.report_interval:]).mean()
                )
                mean_time = int(
                    ((get_ms() - start_ms) / args.report_interval)
                    / batch_size
                )
                progress_clean()
                LOGGER.info(
                    "Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                    batch_num,
                    mean_loss,
                    mean_cost,
                    mean_time,
                )
                if diagnostics is not None:
                    LOGGER.info(
                        "Advice weight norm: %.6f | Advice gradient norm: %.6f",
                        diagnostics["weight_norm"],
                        diagnostics["gradient_norm"],
                    )

                _append_report(
                    history,
                    batch_num,
                    mean_loss,
                    mean_cost,
                    mean_time,
                    diagnostics,
                )
                start_ms = get_ms()

            if (
                args.checkpoint_interval != 0
                and batch_num % args.checkpoint_interval == 0
            ):
                save_checkpoint(model, args, batch_num, history)

    except KeyboardInterrupt:
        history["run"]["status"] = "interrupted"
        history["run"]["last_batch"] = int(batch_num)
        if batch_num > 0:
            save_checkpoint(
                model,
                args,
                batch_num,
                history,
                label="interrupted-batch-{:08d}".format(batch_num),
            )
        else:
            _save_history(history, run_dir)
        LOGGER.warning("Training interrupted at batch %d", batch_num)
        raise
    except Exception:
        history["run"]["status"] = "failed"
        history["run"]["last_batch"] = int(batch_num)
        _save_history(history, run_dir)
        LOGGER.exception("Training failed at batch %d", batch_num)
        raise

    history["run"]["status"] = "completed"
    history["run"]["completed_at"] = _utc_now()
    history["run"]["last_batch"] = int(batch_num)

    if args.save_final and batch_num > 0:
        save_checkpoint(
            model,
            args,
            batch_num,
            history,
            label="final",
        )
    else:
        _save_history(history, run_dir)

    if last_diagnostics is not None:
        LOGGER.info(
            "Final advice weight norm: %.6f",
            last_diagnostics["weight_norm"],
        )
    LOGGER.info("Done training.")
    return history


def init_arguments():
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed value for RNGs",
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default="copy",
        help="Choose the task to train",
    )
    parser.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        help='Override model params, e.g. "-padvice_type=relation"',
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help="Checkpoint interval; use 0 to disable periodic checkpoints",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="./runs/manual",
        help="Dedicated directory for this run",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=REPORT_INTERVAL,
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save checkpoint-final.pt after training",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run identifier stored in history.json",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch behavior where supported",
    )

    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    return parser.parse_args()


def update_model_params(params, update):
    update_dict = {}
    for parameter in update:
        match = re.fullmatch(r"([^=]+)=(.*)", parameter)
        if not match:
            raise ValueError("Unable to parse param update {!r}".format(parameter))
        key, value = match.groups()
        update_dict[key] = value

    valid_names = {field.name for field in attr.fields(type(params))}
    unknown = sorted(set(update_dict) - valid_names)
    if unknown:
        raise ValueError(
            "Unknown model parameters: {}. Valid parameters: {}".format(
                ", ".join(unknown),
                ", ".join(sorted(valid_names)),
            )
        )

    return attr.evolve(params, **update_dict)


def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)
    model_cls, params_cls = TASKS[args.task]
    params = update_model_params(params_cls(), args.param)
    LOGGER.info(params)
    return model_cls(params=params)


def init_logging():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )


def main():
    init_logging()
    args = init_arguments()
    init_seed(args.seed, deterministic=args.deterministic)
    model = init_model(args)
    LOGGER.info(
        "Total number of parameters: %d",
        model.net.calculate_num_params(),
    )
    train_model(model, args)


if __name__ == "__main__":
    main()
