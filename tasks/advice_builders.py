"""Task-specific structural advice builders.

Each builder returns an AdviceProgram consisting of

* a fixed-width read-only matrix A with shape (num_rows, 16);
* a deterministic schedule rho with one row index per model step.

The rows never contain input symbols, target values, labels, or concrete rows
of the writable NTM task memory. Logical positions are normalized to ``[0, 1]``
and remain controller inputs rather than physical memory addresses.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple, Optional

import torch


ADVICE_WIDTH = 16


class AdviceFeature(IntEnum):
    """Fixed feature layout shared by all three tasks."""

    LENGTH_RATIO = 0
    INVERSE_LENGTH = 1
    POSITION = 2
    RELATED_POSITION = 3
    IS_START = 4
    IS_END = 5
    INPUT_PHASE = 6
    OUTPUT_PHASE = 7
    TRANSITION_PHASE = 8
    IDENTITY_RELATION = 9
    SUCCESSOR_RELATION = 10
    MIRROR_RELATION = 11
    CYCLE_BOUNDARY = 12
    STORE_OPERATION = 13
    RECALL_OPERATION = 14
    COMPARE_OPERATION = 15


class AdviceProgram(NamedTuple):
    """Read-only advice matrix and deterministic access schedule."""

    matrix: torch.Tensor
    schedule: torch.Tensor


COMMON_ADVICE_TYPES = {
    "none",
    "zero",
    "random",
    "length",
    "position",
    "relation",
    "operation",
    "combined",
    "wrong",
}

COPY_ADVICE_TYPES = COMMON_ADVICE_TYPES | {"write_only", "read_only"}
REPEAT_COPY_ADVICE_TYPES = COMMON_ADVICE_TYPES | {
    "copy_only",
    "repeat_only",
}
PALINDROME_ADVICE_TYPES = COMMON_ADVICE_TYPES | {
    "count",       # backwards-compatible alias
    "pair_index",  # clearer name for the same representation
}


def uses_advice(advice_type: str) -> bool:
    return normalize_advice_type(advice_type) != "none"


def normalize_advice_type(advice_type: str) -> str:
    if not isinstance(advice_type, str):
        raise TypeError("advice_type must be a string")
    return advice_type.strip().lower().replace("-", "_")


def _validate_type(advice_type: str, valid_types) -> str:
    advice_type = normalize_advice_type(advice_type)
    if advice_type not in valid_types:
        raise ValueError(
            "Unknown advice type {!r}. Valid values are: {}".format(
                advice_type,
                ", ".join(sorted(valid_types)),
            )
        )
    return advice_type


def _length_features(length: int):
    if length <= 0:
        raise ValueError("length must be positive")
    return length / (length + 1.0), 1.0 / (length + 1.0)


def _normalized_position(index: int, length: int) -> float:
    if length <= 1:
        return 0.0
    return float(index) / float(length - 1)


def _row(
    length: int,
    *,
    position: Optional[int] = None,
    related_position: Optional[int] = None,
    input_phase: bool = False,
    output_phase: bool = False,
    transition_phase: bool = False,
    identity_relation: bool = False,
    successor_relation: bool = False,
    mirror_relation: bool = False,
    cycle_boundary: bool = False,
    store_operation: bool = False,
    recall_operation: bool = False,
    compare_operation: bool = False,
    is_start_override: Optional[bool] = None,
    is_end_override: Optional[bool] = None,
) -> torch.Tensor:
    """Construct one semantic advice row."""
    values = torch.zeros(ADVICE_WIDTH, dtype=torch.float32)
    length_ratio, inverse_length = _length_features(length)
    values[AdviceFeature.LENGTH_RATIO] = length_ratio
    values[AdviceFeature.INVERSE_LENGTH] = inverse_length

    if position is not None:
        if position < 0 or position >= length:
            raise ValueError("position is outside the logical sequence")
        values[AdviceFeature.POSITION] = _normalized_position(position, length)
        values[AdviceFeature.IS_START] = float(position == 0)
        values[AdviceFeature.IS_END] = float(position == length - 1)

    if is_start_override is not None:
        values[AdviceFeature.IS_START] = float(is_start_override)
    if is_end_override is not None:
        values[AdviceFeature.IS_END] = float(is_end_override)

    if related_position is not None:
        if related_position < 0 or related_position >= length:
            raise ValueError("related_position is outside the logical sequence")
        values[AdviceFeature.RELATED_POSITION] = _normalized_position(
            related_position,
            length,
        )

    values[AdviceFeature.INPUT_PHASE] = float(input_phase)
    values[AdviceFeature.OUTPUT_PHASE] = float(output_phase)
    values[AdviceFeature.TRANSITION_PHASE] = float(transition_phase)
    values[AdviceFeature.IDENTITY_RELATION] = float(identity_relation)
    values[AdviceFeature.SUCCESSOR_RELATION] = float(successor_relation)
    values[AdviceFeature.MIRROR_RELATION] = float(mirror_relation)
    values[AdviceFeature.CYCLE_BOUNDARY] = float(cycle_boundary)
    values[AdviceFeature.STORE_OPERATION] = float(store_operation)
    values[AdviceFeature.RECALL_OPERATION] = float(recall_operation)
    values[AdviceFeature.COMPARE_OPERATION] = float(compare_operation)
    return values


def _stack(rows):
    if not rows:
        raise ValueError("an advice matrix must contain at least one row")
    return torch.stack(rows, dim=0)


def validate_advice_program(
    program: Optional[AdviceProgram],
    *,
    expected_steps: Optional[int] = None,
) -> Optional[AdviceProgram]:
    """Validate the tensor contract shared by builders, tasks and training."""
    if program is None:
        return None

    matrix, schedule = program
    if not torch.is_tensor(matrix) or not torch.is_tensor(schedule):
        raise TypeError("AdviceProgram entries must be torch tensors")
    if matrix.dim() != 2 or matrix.size(1) != ADVICE_WIDTH:
        raise ValueError(
            "advice matrix must have shape (num_rows, {})".format(
                ADVICE_WIDTH
            )
        )
    if matrix.size(0) <= 0:
        raise ValueError("advice matrix must contain at least one row")
    if not matrix.is_floating_point():
        raise TypeError("advice matrix must use a floating-point dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError("advice matrix contains non-finite values")

    if schedule.dim() != 1:
        raise ValueError("task builders must return a one-dimensional schedule")
    if schedule.dtype != torch.long:
        raise TypeError("advice schedule must use torch.long")
    if schedule.numel() <= 0:
        raise ValueError("advice schedule must contain at least one step")
    if expected_steps is not None and schedule.numel() != int(expected_steps):
        raise ValueError(
            "advice schedule contains {} steps, expected {}".format(
                schedule.numel(),
                int(expected_steps),
            )
        )
    if torch.any(schedule < -1):
        raise IndexError("advice schedule indices must be -1 or non-negative")
    active = schedule.ge(0)
    if active.any() and torch.any(schedule[active] >= matrix.size(0)):
        raise IndexError("advice schedule contains an out-of-range row index")

    return AdviceProgram(
        matrix=matrix.to(dtype=torch.float32),
        schedule=schedule.to(dtype=torch.long),
    )


def _program(matrix, schedule, total_steps: int) -> AdviceProgram:
    return validate_advice_program(
        AdviceProgram(matrix=matrix, schedule=schedule),
        expected_steps=total_steps,
    )


def _matched_random_program(
    reference_matrix: torch.Tensor,
    schedule: torch.Tensor,
    seed: int,
    total_steps: int,
) -> AdviceProgram:
    """Create deterministic sparsity- and norm-matched random advice.

    Each random row has the same number of active entries and the same L2 norm
    as the corresponding structured reference row, but active dimensions and
    values are random. This controls vector scale and density more fairly than
    a dense ``U(0, 1)`` matrix.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.zeros_like(reference_matrix, dtype=torch.float32)

    for row_index, reference_row in enumerate(reference_matrix):
        active_count = int(reference_row.ne(0).sum().item())
        reference_norm = float(reference_row.norm().item())
        if active_count == 0 or reference_norm == 0.0:
            continue

        active_dimensions = torch.randperm(
            ADVICE_WIDTH,
            generator=generator,
        )[:active_count]
        values = torch.rand(
            active_count,
            generator=generator,
            dtype=torch.float32,
        )
        values = values / values.norm().clamp_min(1e-12) * reference_norm
        random_matrix[row_index, active_dimensions] = values

    return _program(
        random_matrix,
        schedule.clone(),
        total_steps,
    )


def build_copy_advice(
    sequence_length: int,
    advice_type: str,
    *,
    random_seed: int = 0,
) -> Optional[AdviceProgram]:
    """Build structural advice for Copy.

    Total model steps are ``n + 1 + n``: input symbols, delimiter and outputs.
    """
    advice_type = _validate_type(advice_type, COPY_ADVICE_TYPES)
    if advice_type == "none":
        return None

    n = int(sequence_length)
    if n <= 0:
        raise ValueError("sequence_length must be positive")
    if advice_type == "wrong" and n < 2:
        raise ValueError("wrong Copy advice requires sequence_length >= 2")

    total_steps = 2 * n + 1

    if advice_type == "zero":
        return _program(
            torch.zeros(1, ADVICE_WIDTH),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    if advice_type == "length":
        return _program(
            _row(n).unsqueeze(0),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    position_rows = [_row(n, position=i) for i in range(n)]
    position_schedule = torch.tensor(
        list(range(n)) + [-1] + list(range(n)),
        dtype=torch.long,
    )

    if advice_type == "position":
        return _program(_stack(position_rows), position_schedule, total_steps)

    relation_rows = [
        _row(
            n,
            position=i,
            related_position=i,
            identity_relation=True,
        )
        for i in range(n)
    ]
    if advice_type == "relation":
        return _program(_stack(relation_rows), position_schedule, total_steps)

    if advice_type == "wrong":
        wrong_rows = [
            _row(
                n,
                position=i,
                related_position=(i + 1) % n,
                identity_relation=True,
            )
            for i in range(n)
        ]
        return _program(_stack(wrong_rows), position_schedule, total_steps)

    if advice_type == "operation":
        matrix = _stack([
            _row(n, input_phase=True, store_operation=True),
            _row(n, transition_phase=True),
            _row(n, output_phase=True, recall_operation=True),
        ])
        schedule = torch.tensor(
            [0] * n + [1] + [2] * n,
            dtype=torch.long,
        )
        return _program(matrix, schedule, total_steps)

    store_rows = [
        _row(
            n,
            position=i,
            related_position=i,
            input_phase=True,
            identity_relation=True,
            store_operation=True,
        )
        for i in range(n)
    ]
    transition_row = _row(n, transition_phase=True)
    recall_rows = [
        _row(
            n,
            position=i,
            related_position=i,
            output_phase=True,
            identity_relation=True,
            recall_operation=True,
        )
        for i in range(n)
    ]
    combined_matrix = _stack(store_rows + [transition_row] + recall_rows)
    combined_schedule = torch.tensor(
        list(range(n))
        + [n]
        + list(range(n + 1, 2 * n + 1)),
        dtype=torch.long,
    )

    if advice_type == "combined":
        return _program(combined_matrix, combined_schedule, total_steps)
    if advice_type == "random":
        return _matched_random_program(
            combined_matrix,
            combined_schedule,
            random_seed,
            total_steps,
        )
    if advice_type == "write_only":
        schedule = torch.tensor(
            list(range(n)) + [-1] + [-1] * n,
            dtype=torch.long,
        )
        return _program(_stack(store_rows), schedule, total_steps)
    if advice_type == "read_only":
        schedule = torch.tensor(
            [-1] * (n + 1) + list(range(n)),
            dtype=torch.long,
        )
        return _program(_stack(recall_rows), schedule, total_steps)

    raise AssertionError("unhandled Copy advice type")


def build_repeat_copy_advice(
    sequence_length: int,
    repetitions: int,
    advice_type: str,
    *,
    random_seed: int = 0,
) -> Optional[AdviceProgram]:
    """Build structural advice for Repeat-Copy.

    Advice rows depend on ``n`` but do not encode ``repetitions``. The task
    runtime uses ``repetitions`` to unroll ``n*r + 1`` output steps. Cyclic
    advice deliberately continues at the final end-marker step; therefore the
    advice itself does not mark the requested stopping point.
    """
    advice_type = _validate_type(advice_type, REPEAT_COPY_ADVICE_TYPES)
    if advice_type == "none":
        return None

    n = int(sequence_length)
    repetitions = int(repetitions)
    if n <= 0:
        raise ValueError("sequence_length must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if advice_type == "wrong" and n < 2:
        raise ValueError(
            "wrong Repeat-Copy advice requires sequence_length >= 2"
        )

    input_steps = n + 2
    output_steps = n * repetitions + 1
    total_steps = input_steps + output_steps

    if advice_type == "zero":
        return _program(
            torch.zeros(1, ADVICE_WIDTH),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    if advice_type == "length":
        return _program(
            _row(n).unsqueeze(0),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    output_positions = [step % n for step in range(output_steps)]
    position_schedule = torch.tensor(
        list(range(n)) + [-1, -1] + output_positions,
        dtype=torch.long,
    )
    position_rows = [_row(n, position=i) for i in range(n)]
    if advice_type == "position":
        return _program(_stack(position_rows), position_schedule, total_steps)

    successor_rows = [
        _row(
            n,
            position=i,
            related_position=(i + 1) % n,
            successor_relation=True,
            cycle_boundary=(i == n - 1),
        )
        for i in range(n)
    ]
    if advice_type == "relation":
        return _program(_stack(successor_rows), position_schedule, total_steps)

    if advice_type == "wrong":
        wrong_rows = [
            _row(
                n,
                position=i,
                related_position=(i + 2) % n,
                successor_relation=True,
                cycle_boundary=(i == n - 1),
            )
            for i in range(n)
        ]
        return _program(_stack(wrong_rows), position_schedule, total_steps)

    if advice_type == "operation":
        matrix = _stack([
            _row(n, input_phase=True, store_operation=True),
            _row(n, transition_phase=True),
            _row(n, output_phase=True, recall_operation=True),
        ])
        schedule = torch.tensor(
            [0] * n + [1, 1] + [2] * output_steps,
            dtype=torch.long,
        )
        return _program(matrix, schedule, total_steps)

    # True Copy-only relation: the same logical position label is presented
    # during input and every repeated output cycle, without phase/operation or
    # successor information.
    if advice_type == "copy_only":
        copy_rows = [
            _row(
                n,
                position=i,
                related_position=i,
                identity_relation=True,
            )
            for i in range(n)
        ]
        schedule = torch.tensor(
            list(range(n)) + [-1, -1] + output_positions,
            dtype=torch.long,
        )
        return _program(_stack(copy_rows), schedule, total_steps)

    if advice_type == "repeat_only":
        schedule = torch.tensor(
            [-1] * input_steps + output_positions,
            dtype=torch.long,
        )
        return _program(_stack(successor_rows), schedule, total_steps)

    store_rows = [
        _row(
            n,
            position=i,
            related_position=i,
            input_phase=True,
            identity_relation=True,
            store_operation=True,
        )
        for i in range(n)
    ]
    transition_row = _row(n, transition_phase=True)
    recall_rows = [
        _row(
            n,
            position=i,
            related_position=(i + 1) % n,
            output_phase=True,
            successor_relation=True,
            cycle_boundary=(i == n - 1),
            recall_operation=True,
        )
        for i in range(n)
    ]
    matrix = _stack(store_rows + [transition_row] + recall_rows)
    output_schedule = [n + 1 + position for position in output_positions]
    schedule = torch.tensor(
        list(range(n)) + [n, n] + output_schedule,
        dtype=torch.long,
    )

    if advice_type == "combined":
        return _program(matrix, schedule, total_steps)
    if advice_type == "random":
        return _matched_random_program(
            matrix,
            schedule,
            random_seed,
            total_steps,
        )

    raise AssertionError("unhandled Repeat-Copy advice type")


def build_even_palindrome_advice(
    sequence_length: int,
    advice_type: str,
    *,
    random_seed: int = 0,
) -> Optional[AdviceProgram]:
    """Build structural advice for even-length palindrome recognition.

    The model receives ``n/2`` post-input processing steps, and only the final
    one is supervised. Mirror rows describe ``(i, n-1-i)`` without symbols or
    labels. ``count`` is kept as a backwards-compatible alias for
    ``pair_index``: it supplies the left logical index of each mirror pair.
    """
    advice_type = _validate_type(advice_type, PALINDROME_ADVICE_TYPES)
    if advice_type == "none":
        return None

    n = int(sequence_length)
    if n <= 0 or n % 2 != 0:
        raise ValueError("sequence_length must be a positive even number")

    comparison_steps = n // 2
    input_steps = n + 1
    total_steps = input_steps + comparison_steps

    if advice_type == "zero":
        return _program(
            torch.zeros(1, ADVICE_WIDTH),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    if advice_type == "length":
        return _program(
            _row(n).unsqueeze(0),
            torch.zeros(total_steps, dtype=torch.long),
            total_steps,
        )

    input_position_rows = [_row(n, position=i) for i in range(n)]
    pair_position_rows = [
        _row(
            n,
            position=i,
            is_start_override=(i == 0),
            is_end_override=(i == comparison_steps - 1),
        )
        for i in range(comparison_steps)
    ]
    position_matrix = _stack(input_position_rows + pair_position_rows)
    position_schedule = torch.tensor(
        list(range(n))
        + [-1]
        + list(range(n, n + comparison_steps)),
        dtype=torch.long,
    )
    if advice_type == "position":
        return _program(position_matrix, position_schedule, total_steps)

    pair_index_rows = [
        _row(
            n,
            position=i,
            output_phase=True,
            compare_operation=True,
            is_start_override=(i == 0),
            is_end_override=(i == comparison_steps - 1),
        )
        for i in range(comparison_steps)
    ]
    if advice_type in {"count", "pair_index"}:
        schedule = torch.tensor(
            [-1] * input_steps + list(range(comparison_steps)),
            dtype=torch.long,
        )
        return _program(_stack(pair_index_rows), schedule, total_steps)

    mirror_rows = [
        _row(
            n,
            position=i,
            related_position=n - 1 - i,
            mirror_relation=True,
            is_start_override=(i == 0),
            is_end_override=(i == comparison_steps - 1),
        )
        for i in range(comparison_steps)
    ]
    relation_matrix = _stack(input_position_rows + mirror_rows)
    relation_schedule = torch.tensor(
        list(range(n))
        + [-1]
        + list(range(n, n + comparison_steps)),
        dtype=torch.long,
    )
    if advice_type == "relation":
        return _program(relation_matrix, relation_schedule, total_steps)

    if advice_type == "wrong":
        wrong_rows = [
            _row(
                n,
                position=i,
                related_position=(n - 2 - i) % n,
                mirror_relation=True,
                is_start_override=(i == 0),
                is_end_override=(i == comparison_steps - 1),
            )
            for i in range(comparison_steps)
        ]
        matrix = _stack(input_position_rows + wrong_rows)
        return _program(matrix, relation_schedule, total_steps)

    if advice_type == "operation":
        matrix = _stack([
            _row(n, input_phase=True, store_operation=True),
            _row(n, transition_phase=True),
            _row(n, output_phase=True, compare_operation=True),
        ])
        schedule = torch.tensor(
            [0] * n + [1] + [2] * comparison_steps,
            dtype=torch.long,
        )
        return _program(matrix, schedule, total_steps)

    store_rows = [
        _row(
            n,
            position=i,
            input_phase=True,
            store_operation=True,
        )
        for i in range(n)
    ]
    transition_row = _row(n, transition_phase=True)
    compare_rows = [
        _row(
            n,
            position=i,
            related_position=n - 1 - i,
            output_phase=True,
            mirror_relation=True,
            compare_operation=True,
            is_start_override=(i == 0),
            is_end_override=(i == comparison_steps - 1),
        )
        for i in range(comparison_steps)
    ]
    matrix = _stack(store_rows + [transition_row] + compare_rows)
    schedule = torch.tensor(
        list(range(n))
        + [n]
        + list(range(n + 1, n + 1 + comparison_steps)),
        dtype=torch.long,
    )

    if advice_type == "combined":
        return _program(matrix, schedule, total_steps)
    if advice_type == "random":
        return _matched_random_program(
            matrix,
            schedule,
            random_seed,
            total_steps,
        )

    raise AssertionError("unhandled Even-Palindrome advice type")
