"""Balanced even-length palindrome-recognition task.

Every sequence has even length and uses the alphabet ``{0, 1}``. Symbols are
one-hot encoded, leaving the all-zero vector for post-input processing. After
the delimiter the NTM receives ``n/2`` zero-input steps, and only the final
processing output is supervised.
"""

from __future__ import annotations

import random

from attr import Factory, attrib, attrs
import torch
from torch import nn, optim

from ntm.aio import EncapsulatedNTM
from .advice_builders import (
    ADVICE_WIDTH,
    PALINDROME_ADVICE_TYPES,
    build_even_palindrome_advice,
    normalize_advice_type,
    uses_advice,
    validate_advice_program,
)


def _choose_even_length(min_len: int, max_len: int, *, py_rng=None) -> int:
    py_rng = random if py_rng is None else py_rng
    even_lengths = [
        length
        for length in range(min_len, max_len + 1)
        if length > 0 and length % 2 == 0
    ]
    if not even_lengths:
        raise ValueError(
            "The palindrome length range must contain a positive even length"
        )
    return py_rng.choice(even_lengths)


def _balanced_labels(
    batch_num: int,
    batch_size: int,
    *,
    torch_generator=None,
) -> torch.Tensor:
    """Return exactly/approximately balanced labels for every batch size."""
    labels = (
        torch.arange(batch_size, dtype=torch.long)
        + int(batch_num)
    ) % 2
    if batch_size > 1:
        labels = labels[
            torch.randperm(
                batch_size,
                generator=torch_generator,
            )
        ]
    return labels


def _generate_sequences(
    sequence_length: int,
    labels: torch.Tensor,
    *,
    py_rng=None,
    torch_generator=None,
) -> torch.Tensor:
    """Generate guaranteed palindromes and guaranteed non-palindromes.

    Negative examples are hard negatives: they begin as palindromes and one
    mirrored pair is made inconsistent. The resulting negative distribution is
    intentionally not uniform over all non-palindromes and must be documented
    as part of the experimental task definition.
    """
    py_rng = random if py_rng is None else py_rng
    batch_size = int(labels.numel())
    half_length = sequence_length // 2

    left_half = torch.randint(
        0,
        2,
        (half_length, batch_size),
        dtype=torch.long,
        generator=torch_generator,
    )
    right_half = torch.flip(left_half, dims=[0]).clone()

    negative_indices = torch.nonzero(
        labels.eq(0),
        as_tuple=False,
    ).flatten()
    for batch_index in negative_indices.tolist():
        pair_index = py_rng.randrange(half_length)
        mirrored_index = half_length - 1 - pair_index
        right_half[mirrored_index, batch_index] = (
            1 - right_half[mirrored_index, batch_index]
        )

    return torch.cat([left_half, right_half], dim=0)


def is_even_palindrome(sequence: torch.Tensor) -> torch.Tensor:
    """Check a tensor of shape ``(length, batch)`` for membership."""
    if sequence.dim() != 2:
        raise ValueError("sequence must have shape (length, batch)")
    if sequence.size(0) % 2 != 0:
        return torch.zeros(sequence.size(1), dtype=torch.bool)
    return sequence.eq(torch.flip(sequence, dims=[0])).all(dim=0)


def dataloader(
    num_batches,
    batch_size,
    min_len,
    max_len,
    *,
    py_rng=None,
    torch_generator=None,
):
    """Yield balanced palindrome-classification batches."""
    for batch_num in range(1, num_batches + 1):
        seq_len = _choose_even_length(
            min_len,
            max_len,
            py_rng=py_rng,
        )
        labels = _balanced_labels(
            batch_num,
            batch_size,
            torch_generator=torch_generator,
        )
        symbols = _generate_sequences(
            seq_len,
            labels,
            py_rng=py_rng,
            torch_generator=torch_generator,
        )

        inp = torch.zeros(seq_len + 1, batch_size, 3)
        inp[:seq_len, :, 0] = symbols.eq(0).float()
        inp[:seq_len, :, 1] = symbols.eq(1).float()
        inp[seq_len, :, 2] = 1.0

        comparison_steps = seq_len // 2
        outp = torch.zeros(comparison_steps, batch_size, 1)
        outp[-1, :, 0] = labels.float()

        loss_mask = torch.zeros_like(outp)
        loss_mask[-1, :, 0] = 1.0

        metadata = {
            "batch_num": batch_num,
            "sequence_length": seq_len,
            "input_steps": seq_len + 1,
            "output_steps": comparison_steps,
            "positive_examples": int(labels.sum().item()),
            "negative_examples": int(labels.numel() - labels.sum().item()),
            "loss_mask": loss_mask,
        }
        yield batch_num, inp, outp, metadata


@attrs
class EvenPalindromeTaskParams(object):
    name = attrib(default="even-palindrome-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)

    # Backwards compatibility: num_heads denotes the number of read heads.
    # Palindrome comparison uses two reads but only one write.
    num_heads = attrib(default=2, converter=int)
    num_write_heads = attrib(default=1, converter=int)

    sequence_min_len = attrib(default=2, converter=int)
    sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=100000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)

    advice_type = attrib(default="none", converter=normalize_advice_type)
    advice_size = attrib(default=ADVICE_WIDTH, converter=int)
    advice_strength = attrib(default=1.0, converter=float)
    advice_random_seed = attrib(default=3141, converter=int)


@attrs
class EvenPalindromeTaskModelTraining(object):
    params = attrib(default=Factory(EvenPalindromeTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        if self.params.advice_type not in PALINDROME_ADVICE_TYPES:
            raise ValueError(
                "Invalid Even-Palindrome advice_type {!r}".format(
                    self.params.advice_type
                )
            )
        if (
            uses_advice(self.params.advice_type)
            and self.params.advice_size != ADVICE_WIDTH
        ):
            raise ValueError(
                "The current builders require advice_size={}, got {}".format(
                    ADVICE_WIDTH,
                    self.params.advice_size,
                )
            )

        advice_size = (
            self.params.advice_size
            if uses_advice(self.params.advice_type)
            else 0
        )
        return EncapsulatedNTM(
            num_inputs=3,
            num_outputs=1,
            controller_size=self.params.controller_size,
            controller_layers=self.params.controller_layers,
            num_heads=self.params.num_heads,
            N=self.params.memory_n,
            M=self.params.memory_m,
            advice_size=advice_size,
            advice_strength=self.params.advice_strength,
            num_read_heads=self.params.num_heads,
            num_write_heads=self.params.num_write_heads,
        )

    def make_dataloader(self, *, num_batches=None, seed=None):
        count = (
            self.params.num_batches
            if num_batches is None
            else int(num_batches)
        )
        if seed is None:
            py_rng = None
            torch_generator = None
        else:
            py_rng = random.Random(int(seed))
            torch_generator = torch.Generator(device="cpu")
            torch_generator.manual_seed(int(seed))

        return dataloader(
            count,
            self.params.batch_size,
            self.params.sequence_min_len,
            self.params.sequence_max_len,
            py_rng=py_rng,
            torch_generator=torch_generator,
        )

    @dataloader.default
    def default_dataloader(self):
        return self.make_dataloader()

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(
            self.net.parameters(),
            momentum=self.params.rmsprop_momentum,
            alpha=self.params.rmsprop_alpha,
            lr=self.params.rmsprop_lr,
        )

    def build_advice(self, metadata):
        program = build_even_palindrome_advice(
            metadata["sequence_length"],
            self.params.advice_type,
            random_seed=(
                self.params.advice_random_seed
                + 1009 * int(metadata["sequence_length"])
            ),
        )
        return validate_advice_program(
            program,
            expected_steps=(
                int(metadata["input_steps"])
                + int(metadata["output_steps"])
            ),
        )
