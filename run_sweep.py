#!/usr/bin/env python
"""Run a reproducible grid of NTM advice types and random seeds.

Typical use from the repository root:

    python run_sweep.py --config experiments/copy_sweep.json

Runs are sequential by default, which is appropriate for one GPU. Each run gets
its own directory and canonical ``history.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List


MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    required = ["task", "advices", "seeds", "num_batches", "output_root"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing sweep config keys: {}".format(", ".join(missing)))

    task = str(config["task"])
    if task not in {"copy", "repeat-copy", "even-palindrome"}:
        raise ValueError("Unsupported task: {}".format(task))

    advices = [str(value).strip().lower().replace("-", "_") for value in config["advices"]]
    if not advices:
        raise ValueError("advices must not be empty")

    seeds = [int(value) for value in config["seeds"]]
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds contains duplicates")

    normalized = dict(config)
    normalized["task"] = task
    normalized["advices"] = advices
    normalized["seeds"] = seeds
    normalized["num_batches"] = int(config["num_batches"])
    normalized["output_root"] = str(config["output_root"])
    normalized["report_interval"] = int(config.get("report_interval", 200))
    normalized["checkpoint_interval"] = int(config.get("checkpoint_interval", 5000))
    normalized["save_final"] = bool(config.get("save_final", True))
    normalized["deterministic"] = bool(config.get("deterministic", False))
    normalized["params"] = dict(config.get("params", {}))
    normalized["train_script"] = str(config.get("train_script", "train.py"))
    return normalized


def is_completed(history_path: Path, requested_batches: int) -> bool:
    if not history_path.exists():
        return False
    try:
        history = load_json(history_path)
    except (OSError, json.JSONDecodeError):
        return False

    run = history.get("run", {})
    if run.get("status") != "completed":
        return False
    return int(run.get("last_batch", 0)) >= int(requested_batches)


def build_command(
    config: Dict[str, Any],
    *,
    advice: str,
    seed: int,
    run_dir: Path,
    run_id: str,
) -> List[str]:
    command = [
        sys.executable,
        config["train_script"],
        "--task",
        config["task"],
        "--seed",
        str(seed),
        "-padvice_type={}".format(advice),
        "-pnum_batches={}".format(config["num_batches"]),
        "--report-interval",
        str(config["report_interval"]),
        "--checkpoint-interval",
        str(config["checkpoint_interval"]),
        "--checkpoint-path",
        str(run_dir),
        "--run-id",
        run_id,
    ]

    for key, value in sorted(config["params"].items()):
        # advice_type and num_batches are controlled by the sweep grid.
        if key in {"advice_type", "num_batches"}:
            continue
        command.append("-p{}={}".format(key, value))

    if config["save_final"]:
        command.append("--save-final")
    if config["deterministic"]:
        command.append("--deterministic")
    return command


def tee_process(command: List[str], log_path: Path, prefix: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print("{} {}".format(prefix, line), end="")
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def create_manifest(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    runs = []
    for advice in config["advices"]:
        for seed in config["seeds"]:
            run_id = "{}-{}-seed-{}".format(config["task"], advice, seed)
            runs.append(
                {
                    "run_id": run_id,
                    "task": config["task"],
                    "advice_type": advice,
                    "seed": seed,
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "return_code": None,
                    "run_directory": str(
                        Path(config["output_root"])
                        / config["task"]
                        / advice
                        / "seed_{}".format(seed)
                    ),
                }
            )

    now = utc_now()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "config_file": str(config_path),
        "config": config,
        "runs": runs,
    }


def update_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)


def format_command(command: Iterable[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sweep JSON configuration")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run again even when a completed history.json exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later runs if one training process fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = validate_config(load_json(config_path))

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sweep_manifest.json"

    manifest = create_manifest(config, config_path)
    update_manifest(manifest_path, manifest)

    total = len(manifest["runs"])
    for index, run in enumerate(manifest["runs"], start=1):
        run_dir = Path(run["run_directory"])
        history_path = run_dir / "history.json"
        prefix = "[{}/{} {} seed={}]".format(
            index,
            total,
            run["advice_type"],
            run["seed"],
        )

        if not args.force and is_completed(history_path, config["num_batches"]):
            run["status"] = "skipped_completed"
            run["finished_at"] = utc_now()
            update_manifest(manifest_path, manifest)
            print("{} already completed; skipping.".format(prefix))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            config,
            advice=run["advice_type"],
            seed=run["seed"],
            run_dir=run_dir,
            run_id=run["run_id"],
        )
        command_text = format_command(command)
        run["command"] = command_text

        print("{} {}".format(prefix, command_text))
        if args.dry_run:
            run["status"] = "dry_run"
            update_manifest(manifest_path, manifest)
            continue

        run["status"] = "running"
        run["started_at"] = utc_now()
        update_manifest(manifest_path, manifest)

        return_code = tee_process(
            command,
            run_dir / "console.log",
            prefix,
        )
        run["return_code"] = int(return_code)
        run["finished_at"] = utc_now()
        run["status"] = "completed" if return_code == 0 else "failed"
        update_manifest(manifest_path, manifest)

        if return_code != 0 and not args.continue_on_error:
            print("{} failed with return code {}.".format(prefix, return_code))
            return return_code

    print("Sweep finished. Manifest: {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
