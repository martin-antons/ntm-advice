from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ntm.aio import EncapsulatedNTM
from tasks.advice_builders import (
    ADVICE_WIDTH,
    AdviceFeature,
    COPY_ADVICE_TYPES,
    PALINDROME_ADVICE_TYPES,
    REPEAT_COPY_ADVICE_TYPES,
    build_copy_advice,
    build_even_palindrome_advice,
    build_repeat_copy_advice,
)
from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import (
    RepeatCopyTaskModelTraining,
    RepeatCopyTaskParams,
)
from tasks.evenpalindrometask import (
    EvenPalindromeTaskModelTraining,
    EvenPalindromeTaskParams,
    is_even_palindrome,
)
from train import train_batch


def test_advice_columns_start_at_zero():
    torch.manual_seed(11)
    model = EncapsulatedNTM(
        num_inputs=3,
        num_outputs=2,
        controller_size=8,
        controller_layers=1,
        num_heads=1,
        N=6,
        M=4,
        advice_size=ADVICE_WIDTH,
    )
    weight = model.ntm.controller.lstm.weight_ih_l0.detach()
    assert torch.count_nonzero(weight[:, -ADVICE_WIDTH:]) == 0
    assert torch.count_nonzero(weight[:, :-ADVICE_WIDTH]) > 0


def test_palindrome_uses_two_reads_and_one_write():
    model = EvenPalindromeTaskModelTraining(
        params=EvenPalindromeTaskParams(
            controller_size=8,
            memory_n=8,
            memory_m=4,
            num_batches=1,
            batch_size=2,
            sequence_min_len=4,
            sequence_max_len=4,
            advice_type="combined",
        )
    )
    assert model.net.num_read_heads == 2
    assert model.net.num_write_heads == 1
    assert model.net.ntm.num_read_heads == 2
    assert model.net.ntm.num_write_heads == 1


def test_all_reads_use_the_pre_write_memory():
    torch.manual_seed(13)
    model = EncapsulatedNTM(
        num_inputs=3,
        num_outputs=1,
        controller_size=8,
        controller_layers=1,
        num_heads=2,
        N=7,
        M=4,
        num_read_heads=2,
        num_write_heads=1,
    )
    model.init_sequence(batch_size=2, record_ntm_trace=True)
    model(torch.randn(2, 3))
    trace = model.ntm_trace[0]

    memory_before = trace["memory_before"]
    weightings = trace["read_weightings"]
    expected_reads = torch.matmul(
        weightings.unsqueeze(2),
        memory_before.unsqueeze(1),
    ).squeeze(2)
    assert torch.allclose(
        expected_reads,
        trace["read_vectors"],
        atol=1e-6,
    )


def test_repeat_copy_copy_only_is_a_true_identity_relation():
    program = build_repeat_copy_advice(3, 2, "copy_only")
    assert program.matrix.shape == (3, ADVICE_WIDTH)
    assert torch.all(program.matrix[:, AdviceFeature.IDENTITY_RELATION] == 1)
    assert torch.all(program.matrix[:, AdviceFeature.SUCCESSOR_RELATION] == 0)
    assert torch.all(program.matrix[:, AdviceFeature.INPUT_PHASE] == 0)
    assert torch.all(program.matrix[:, AdviceFeature.OUTPUT_PHASE] == 0)
    assert torch.all(program.matrix[:, AdviceFeature.STORE_OPERATION] == 0)
    assert torch.all(program.matrix[:, AdviceFeature.RECALL_OPERATION] == 0)
    assert program.schedule.tolist() == [0, 1, 2, -1, -1, 0, 1, 2, 0, 1, 2, 0]


def test_random_advice_is_fixed_for_length_and_matched_to_combined():
    combined = build_copy_advice(5, "combined")
    random_a = build_copy_advice(5, "random", random_seed=123)
    random_b = build_copy_advice(5, "random", random_seed=123)
    random_c = build_copy_advice(5, "random", random_seed=124)

    assert torch.equal(random_a.matrix, random_b.matrix)
    assert not torch.equal(random_a.matrix, random_c.matrix)
    assert torch.equal(random_a.schedule, combined.schedule)
    assert torch.equal(
        random_a.matrix.ne(0).sum(dim=1),
        combined.matrix.ne(0).sum(dim=1),
    )
    assert torch.allclose(
        random_a.matrix.norm(dim=1),
        combined.matrix.norm(dim=1),
        atol=1e-6,
    )


def test_task_random_builder_does_not_depend_on_batch_number():
    model = CopyTaskModelTraining(
        params=CopyTaskParams(
            controller_size=8,
            memory_n=8,
            memory_m=4,
            num_batches=1,
            sequence_min_len=3,
            sequence_max_len=3,
            advice_type="random",
        )
    )
    base = {
        "sequence_length": 3,
        "input_steps": 4,
        "output_steps": 3,
    }
    first = model.build_advice({"batch_num": 1, **base})
    later = model.build_advice({"batch_num": 999, **base})
    assert torch.equal(first.matrix, later.matrix)
    assert torch.equal(first.schedule, later.schedule)


@pytest.mark.parametrize("advice_type", sorted(COPY_ADVICE_TYPES))
def test_copy_all_schedules_match_task(advice_type):
    if advice_type == "wrong":
        n = 3
    else:
        n = 2
    program = build_copy_advice(n, advice_type)
    if program is not None:
        assert len(program.schedule) == 2 * n + 1


@pytest.mark.parametrize("advice_type", sorted(REPEAT_COPY_ADVICE_TYPES))
def test_repeat_all_schedules_match_task(advice_type):
    n, repetitions = 3, 2
    program = build_repeat_copy_advice(n, repetitions, advice_type)
    if program is not None:
        assert len(program.schedule) == n + 2 + n * repetitions + 1


@pytest.mark.parametrize("advice_type", sorted(PALINDROME_ADVICE_TYPES))
def test_palindrome_all_schedules_match_task(advice_type):
    n = 6
    program = build_even_palindrome_advice(n, advice_type)
    if program is not None:
        assert len(program.schedule) == n + 1 + n // 2


def test_wrong_advice_rejects_degenerate_length_one():
    with pytest.raises(ValueError):
        build_copy_advice(1, "wrong")
    with pytest.raises(ValueError):
        build_repeat_copy_advice(1, 3, "wrong")


def test_palindrome_data_are_balanced_and_labels_are_correct():
    model = EvenPalindromeTaskModelTraining(
        params=EvenPalindromeTaskParams(
            controller_size=8,
            memory_n=8,
            memory_m=4,
            num_batches=10,
            batch_size=8,
            sequence_min_len=4,
            sequence_max_len=8,
        )
    )
    positives = 0
    negatives = 0
    for _, x, y, metadata in model.make_dataloader(
        num_batches=10,
        seed=777,
    ):
        symbols = x[:-1, :, 1].long()
        labels = y[-1, :, 0].long()
        actual = is_even_palindrome(symbols).long()
        assert torch.equal(labels, actual)
        positives += metadata["positive_examples"]
        negatives += metadata["negative_examples"]
        assert metadata["loss_mask"].sum().item() == x.size(1)
    assert positives == negatives


@pytest.mark.parametrize(
    "model",
    [
        CopyTaskModelTraining(
            params=CopyTaskParams(
                controller_size=8,
                memory_n=8,
                memory_m=4,
                num_batches=1,
                batch_size=2,
                sequence_width=3,
                sequence_min_len=3,
                sequence_max_len=3,
                advice_type="combined",
            )
        ),
        RepeatCopyTaskModelTraining(
            params=RepeatCopyTaskParams(
                controller_size=8,
                memory_n=10,
                memory_m=4,
                num_batches=1,
                batch_size=2,
                sequence_width=3,
                sequence_min_len=3,
                sequence_max_len=3,
                repeat_min=2,
                repeat_max=2,
                advice_type="combined",
            )
        ),
        EvenPalindromeTaskModelTraining(
            params=EvenPalindromeTaskParams(
                controller_size=8,
                memory_n=10,
                memory_m=4,
                num_batches=1,
                batch_size=2,
                sequence_min_len=4,
                sequence_max_len=4,
                advice_type="combined",
            )
        ),
    ],
)
def test_one_optimizer_step_for_each_task(model):
    _, x, y, metadata = next(iter(model.dataloader))
    program = model.build_advice(metadata)
    loss, metrics, diagnostics = train_batch(
        model.net,
        model.criterion,
        model.optimizer,
        x,
        y,
        advice_program=program,
        loss_mask=metadata.get("loss_mask"),
        return_metrics=True,
    )
    assert np.isfinite(loss)
    assert np.isfinite(metrics["cost"])
    assert 0.0 <= metrics["bit_error_rate"] <= 1.0
    assert 0.0 <= metrics["exact_sequence_accuracy"] <= 1.0
    assert diagnostics is not None
    assert diagnostics["weight_norm"] > 0.0
