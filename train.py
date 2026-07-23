#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training entry point for baseline and advice-augmented NTM tasks.

The script writes one canonical ``history.json`` per run. Model checkpoints are
separate ``.pt`` files. Full metric histories are not duplicated into every
checkpoint unless ``--embed-history-in-checkpoint`` is explicitly requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional

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
HISTORY_SCHEMA_VERSION = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_ms() -> float:
    return time.perf_counter() * 1000.0


def _git_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def init_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    if seed is None:
        seed = int(time.time())

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _environment_metadata(device: torch.device) -> Dict[str, Any]:
    cuda_name = None
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)

    slurm_keys = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda_device_name": cuda_name,
        "git_commit": _git_commit(),
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
    }


def _new_history(model, args, device: torch.device) -> Dict[str, Any]:
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
            "validation_interval": int(args.validation_interval),
            "parameter_count": int(model.net.calculate_num_params()),
            "deterministic": bool(args.deterministic),
            "best_validation": None,
        },
        "model_params": model_params,
        "training_args": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list)) or value is None
        },
        "environment": _environment_metadata(device),
        "metrics": {
            "batch": [],
            "loss": [],
            "cost": [],
            "bit_error_rate": [],
            "exact_sequence_accuracy": [],
            "sequence_length": [],
            "repetitions": [],
            "input_steps": [],
            "output_steps": [],
            "batch_time_ms": [],
        },
        "reports": {
            "batch": [],
            "mean_loss": [],
            "mean_cost": [],
            "mean_bit_error_rate": [],
            "mean_exact_sequence_accuracy": [],
            "ms_per_sequence": [],
            # Kept for backwards-compatible plotting: these are the last values
            # in each report interval.
            "advice_weight_norm": [],
            "advice_gradient_norm": [],
            # Explicit interval means for the new schema.
            "mean_advice_weight_norm": [],
            "mean_advice_gradient_norm": [],
        },
        "validation": {
            "batch": [],
            "mean_loss": [],
            "mean_cost": [],
            "mean_bit_error_rate": [],
            "mean_exact_sequence_accuracy": [],
            "num_batches": int(args.validation_batches),
            "seed": int(args.validation_seed),
        },
    }


def _save_history(history: Dict[str, Any], run_dir: Path) -> None:
    history["run"]["updated_at"] = _utc_now()
    _atomic_write_json(run_dir / "history.json", history)


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _last_values(history: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for section_name in ("metrics", "reports", "validation"):
        section = history[section_name]
        result[section_name] = {
            key: (values[-1] if isinstance(values, list) and values else None)
            for key, values in section.items()
            if key not in {"num_batches", "seed"}
        }
    return result


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
        "run": dict(history["run"]),
        "model_params": dict(history["model_params"]),
        "training_args": dict(history["training_args"]),
        "environment": dict(history["environment"]),
        "model_state_dict": model.net.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "rng_state": _rng_state(),
        "history_path": "history.json",
        "last_values": _last_values(history),
    }

    if args.embed_history_in_checkpoint:
        checkpoint["metrics"] = history["metrics"]
        checkpoint["reports"] = history["reports"]
        checkpoint["validation"] = history["validation"]

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


def _compute_prediction_metrics(
    prediction,
    target,
    loss_mask=None,
) -> Dict[str, float]:
    binary = prediction.detach().ge(0.5).to(target.dtype)
    errors = binary.ne(target.detach()).to(prediction.dtype)

    if loss_mask is None:
        active = torch.ones_like(errors)
    else:
        active = loss_mask.to(
            device=errors.device,
            dtype=errors.dtype,
        )
        if active.shape != errors.shape:
            raise ValueError("loss_mask shape does not match prediction shape")
        errors = errors * active

    wrong_bits = errors.sum()
    supervised_bits = active.sum().clamp_min(1.0)
    errors_per_sequence = errors.sum(dim=(0, 2))

    return {
        # Backwards-compatible cost: wrong supervised bits per sequence.
        "cost": float(wrong_bits.item() / prediction.size(1)),
        "bit_error_rate": float(
            (wrong_bits / supervised_bits).item()
        ),
        "exact_sequence_accuracy": float(
            errors_per_sequence.eq(0).float().mean().item()
        ),
    }


def _check_schedule_contract(advice_program, expected_steps: int) -> None:
    if advice_program is None:
        return
    actual_steps = int(advice_program.schedule.size(0))
    if actual_steps != int(expected_steps):
        raise ValueError(
            "Advice schedule contains {} steps, but the task executes {}".format(
                actual_steps,
                int(expected_steps),
            )
        )


def _assert_schedule_consumed(net) -> None:
    if (
        getattr(net, "uses_advice", False)
        and net.advice_reader.is_loaded
        and net.time_step != net.advice_reader.total_steps
    ):
        raise RuntimeError(
            "The task consumed {} model steps, but the advice schedule has {}"
            .format(net.time_step, net.advice_reader.total_steps)
        )


def _init_model_sequence(
    net,
    batch_size,
    advice_program=None,
    *,
    record_advice_trace=False,
    record_ntm_trace=False,
):
    if advice_program is None:
        net.init_sequence(
            batch_size,
            record_advice_trace=record_advice_trace,
            record_ntm_trace=record_ntm_trace,
        )
    else:
        net.init_sequence(
            batch_size,
            advice=advice_program.matrix,
            advice_schedule=advice_program.schedule,
            record_advice_trace=record_advice_trace,
            record_ntm_trace=record_ntm_trace,
        )


def _advice_diagnostics(net) -> Optional[Dict[str, float]]:
    """Inspect the advice columns in the first LSTM input matrix."""
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
    return_metrics=False,
):
    """Run one optimizer step.

    The default return value remains ``(loss, cost, diagnostics)`` for backwards
    compatibility. ``return_metrics=True`` returns the full metric dictionary
    in place of the scalar cost.
    """
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()
    expected_steps = inp_seq_len + outp_seq_len
    _check_schedule_contract(advice_program, expected_steps)

    _init_model_sequence(net, batch_size, advice_program)

    for step in range(inp_seq_len):
        net(X[step])

    y_out = Y.new_zeros(Y.size())
    for step in range(outp_seq_len):
        y_out[step], _ = net()

    _assert_schedule_consumed(net)

    loss = _compute_loss(criterion, y_out, Y, loss_mask)
    loss.backward()
    clip_grads(net)

    diagnostics = _advice_diagnostics(net)
    optimizer.step()

    if diagnostics is not None:
        updated = _advice_diagnostics(net)
        diagnostics["weight_norm"] = updated["weight_norm"]

    prediction_metrics = _compute_prediction_metrics(
        y_out,
        Y,
        loss_mask,
    )
    if return_metrics:
        return loss.item(), prediction_metrics, diagnostics
    return loss.item(), prediction_metrics["cost"], diagnostics


def evaluate(
    net,
    criterion,
    X,
    Y,
    *,
    advice_program=None,
    loss_mask=None,
    record_advice_trace=False,
    record_ntm_trace=False,
    collect_states=True,
    return_outputs=True,
):
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()
    expected_steps = inp_seq_len + outp_seq_len
    _check_schedule_contract(advice_program, expected_steps)

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            _init_model_sequence(
                net,
                batch_size,
                advice_program,
                record_advice_trace=record_advice_trace,
                record_ntm_trace=record_ntm_trace,
            )

            states = []
            for step in range(inp_seq_len):
                _, state = net(X[step])
                if collect_states:
                    states.append(state)

            y_out = Y.new_zeros(Y.size())
            for step in range(outp_seq_len):
                y_out[step], state = net()
                if collect_states:
                    states.append(state)

            _assert_schedule_consumed(net)
            loss = _compute_loss(criterion, y_out, Y, loss_mask)
            prediction_metrics = _compute_prediction_metrics(
                y_out,
                Y,
                loss_mask,
            )
    finally:
        net.train(was_training)

    result = {
        "loss": loss.item(),
        **prediction_metrics,
        "states": states,
        "advice_trace": list(net.advice_trace),
        "ntm_trace": list(net.ntm_trace),
    }
    if return_outputs:
        result["y_out"] = y_out
        result["y_out_binarized"] = y_out.ge(0.5).to(Y.dtype)
    return result


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
    mean_bit_error_rate: float,
    mean_exact_sequence_accuracy: float,
    ms_per_sequence: int,
    diagnostics: Optional[Dict[str, float]],
    mean_weight_norm: Optional[float],
    mean_gradient_norm: Optional[float],
) -> None:
    reports = history["reports"]
    reports["batch"].append(int(batch_num))
    reports["mean_loss"].append(float(mean_loss))
    reports["mean_cost"].append(float(mean_cost))
    reports["mean_bit_error_rate"].append(float(mean_bit_error_rate))
    reports["mean_exact_sequence_accuracy"].append(
        float(mean_exact_sequence_accuracy)
    )
    reports["ms_per_sequence"].append(int(ms_per_sequence))
    reports["advice_weight_norm"].append(
        None if diagnostics is None else float(diagnostics["weight_norm"])
    )
    reports["advice_gradient_norm"].append(
        None if diagnostics is None else float(diagnostics["gradient_norm"])
    )
    reports["mean_advice_weight_norm"].append(mean_weight_norm)
    reports["mean_advice_gradient_norm"].append(mean_gradient_norm)


def _prepare_validation_batches(model, args):
    if args.validation_interval <= 0 or args.validation_batches <= 0:
        return []
    if not hasattr(model, "make_dataloader"):
        raise RuntimeError(
            "Validation was requested, but the task has no make_dataloader()"
        )
    return list(
        model.make_dataloader(
            num_batches=args.validation_batches,
            seed=args.validation_seed,
        )
    )


def _run_validation(model, validation_batches, device) -> Dict[str, float]:
    values = {
        "loss": [],
        "cost": [],
        "bit_error_rate": [],
        "exact_sequence_accuracy": [],
    }
    for batch in validation_batches:
        _, x, y, metadata = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        advice_program = _build_advice(model, metadata)
        result = evaluate(
            model.net,
            model.criterion,
            x,
            y,
            advice_program=advice_program,
            loss_mask=metadata.get("loss_mask"),
            collect_states=False,
            return_outputs=False,
        )
        for key in values:
            values[key].append(float(result[key]))

    return {
        "mean_loss": float(np.mean(values["loss"])),
        "mean_cost": float(np.mean(values["cost"])),
        "mean_bit_error_rate": float(
            np.mean(values["bit_error_rate"])
        ),
        "mean_exact_sequence_accuracy": float(
            np.mean(values["exact_sequence_accuracy"])
        ),
    }


def _is_better_validation(candidate, best) -> bool:
    if best is None:
        return True
    candidate_accuracy = candidate["mean_exact_sequence_accuracy"]
    best_accuracy = best["mean_exact_sequence_accuracy"]
    if candidate_accuracy > best_accuracy + 1e-12:
        return True
    if abs(candidate_accuracy - best_accuracy) <= 1e-12:
        return candidate["mean_loss"] < best["mean_loss"]
    return False


def _record_validation(
    model,
    args,
    history,
    validation_batches,
    device,
    batch_num,
):
    result = _run_validation(model, validation_batches, device)
    validation = history["validation"]
    validation["batch"].append(int(batch_num))
    for key, value in result.items():
        validation[key].append(float(value))

    LOGGER.info(
        "Validation at batch %d: loss=%.6f BER=%.6f exact=%.4f",
        batch_num,
        result["mean_loss"],
        result["mean_bit_error_rate"],
        result["mean_exact_sequence_accuracy"],
    )

    current = {"batch": int(batch_num), **result}
    best = history["run"]["best_validation"]
    if _is_better_validation(current, best):
        history["run"]["best_validation"] = current
        save_checkpoint(
            model,
            args,
            batch_num,
            history,
            label="best",
        )
    return result


def train_model(model, args, device: torch.device):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    run_dir = Path(args.checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Training model for %d batches (batch_size=%d) on %s...",
        num_batches,
        batch_size,
        device,
    )

    history = _new_history(model, args, device)
    _save_history(history, run_dir)
    validation_batches = _prepare_validation_batches(model, args)

    report_start_ms = get_ms()
    batch_num = 0
    last_diagnostics = None
    interval_weight_norms = []
    interval_gradient_norms = []

    try:
        for batch in model.dataloader:
            batch_num, x, y, metadata = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            advice_program = _build_advice(model, metadata)
            loss_mask = metadata.get("loss_mask")

            batch_start_ms = get_ms()
            loss, prediction_metrics, diagnostics = train_batch(
                model.net,
                model.criterion,
                model.optimizer,
                x,
                y,
                advice_program=advice_program,
                loss_mask=loss_mask,
                return_metrics=True,
            )
            batch_time_ms = get_ms() - batch_start_ms
            last_diagnostics = diagnostics

            if diagnostics is not None:
                interval_weight_norms.append(diagnostics["weight_norm"])
                interval_gradient_norms.append(diagnostics["gradient_norm"])

            metrics = history["metrics"]
            metrics["batch"].append(int(batch_num))
            metrics["loss"].append(float(loss))
            metrics["cost"].append(float(prediction_metrics["cost"]))
            metrics["bit_error_rate"].append(
                float(prediction_metrics["bit_error_rate"])
            )
            metrics["exact_sequence_accuracy"].append(
                float(prediction_metrics["exact_sequence_accuracy"])
            )
            metrics["sequence_length"].append(
                int(metadata.get("sequence_length", y.size(0)))
            )
            repetitions = metadata.get("repetitions")
            metrics["repetitions"].append(
                None if repetitions is None else int(repetitions)
            )
            metrics["input_steps"].append(
                int(metadata.get("input_steps", x.size(0)))
            )
            metrics["output_steps"].append(
                int(metadata.get("output_steps", y.size(0)))
            )
            metrics["batch_time_ms"].append(float(batch_time_ms))
            history["run"]["last_batch"] = int(batch_num)

            progress_bar(batch_num, args.report_interval, loss)

            if batch_num % args.report_interval == 0:
                interval = slice(-args.report_interval, None)
                mean_loss = float(np.mean(metrics["loss"][interval]))
                mean_cost = float(np.mean(metrics["cost"][interval]))
                mean_ber = float(
                    np.mean(metrics["bit_error_rate"][interval])
                )
                mean_exact = float(
                    np.mean(metrics["exact_sequence_accuracy"][interval])
                )
                mean_time = int(
                    ((get_ms() - report_start_ms) / args.report_interval)
                    / batch_size
                )
                mean_weight = (
                    float(np.mean(interval_weight_norms))
                    if interval_weight_norms
                    else None
                )
                mean_gradient = (
                    float(np.mean(interval_gradient_norms))
                    if interval_gradient_norms
                    else None
                )

                progress_clean()
                LOGGER.info(
                    "Batch %d Loss: %.6f Cost: %.2f BER: %.6f "
                    "Exact: %.4f Time: %d ms/sequence",
                    batch_num,
                    mean_loss,
                    mean_cost,
                    mean_ber,
                    mean_exact,
                    mean_time,
                )
                if diagnostics is not None:
                    LOGGER.info(
                        "Advice weight norm: %.6f | gradient norm: %.6f",
                        diagnostics["weight_norm"],
                        diagnostics["gradient_norm"],
                    )

                _append_report(
                    history,
                    batch_num,
                    mean_loss,
                    mean_cost,
                    mean_ber,
                    mean_exact,
                    mean_time,
                    diagnostics,
                    mean_weight,
                    mean_gradient,
                )
                report_start_ms = get_ms()
                interval_weight_norms = []
                interval_gradient_norms = []
                _save_history(history, run_dir)

            if (
                validation_batches
                and batch_num % args.validation_interval == 0
            ):
                _record_validation(
                    model,
                    args,
                    history,
                    validation_batches,
                    device,
                    batch_num,
                )

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

    if (
        validation_batches
        and (
            not history["validation"]["batch"]
            or history["validation"]["batch"][-1] != batch_num
        )
    ):
        _record_validation(
            model,
            args,
            history,
            validation_batches,
            device,
            batch_num,
        )

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
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' uses CUDA when available.",
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
        "--validation-interval",
        type=int,
        default=0,
        help="Run fixed validation every N batches; 0 disables validation",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=424242,
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save checkpoint-final.pt after training",
    )
    parser.add_argument(
        "--embed-history-in-checkpoint",
        action="store_true",
        help="Embed full metric arrays in every checkpoint (legacy/heavy)",
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



def validate_runtime_args(args) -> None:
    if args.report_interval <= 0:
        raise ValueError("--report-interval must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")
    if args.validation_interval < 0:
        raise ValueError("--validation-interval must be non-negative")
    if args.validation_batches < 0:
        raise ValueError("--validation-batches must be non-negative")
    if args.validation_interval > 0 and args.validation_batches == 0:
        raise ValueError(
            "--validation-batches must be positive when validation is enabled"
        )


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
    validate_runtime_args(args)
    init_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    model = init_model(args)
    model.net.to(device)
    LOGGER.info(
        "Total number of parameters: %d",
        model.net.calculate_num_params(),
    )
    train_model(model, args, device)


if __name__ == "__main__":
    main()
