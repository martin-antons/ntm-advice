"""Balanced even-length palindrome-recognition task.

Every input sequence has an even length and uses the alphabet {0, 1}. Symbols
are one-hot encoded, so the all-zero vector remains reserved for post-input
processing steps. Positive and negative examples are generated in balanced form.

After the delimiter, the NTM receives ``n/2`` zero-input processing steps. The
classification loss is applied only to the last of those steps. This gives both a
baseline NTM and an advice-augmented NTM enough recurrent computation time to
retrieve and compare stored symbols.
"""

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
)


def _choose_even_length(min_len: int, max_len: int) -> int:
    even_lengths = [
        length
        for length in range(min_len, max_len + 1)
        if length > 0 and length % 2 == 0
    ]
    if not even_lengths:
        raise ValueError(
            "The palindrome length range must contain a positive even length"
        )
    return random.choice(even_lengths)


def _balanced_labels(batch_num: int, batch_size: int) -> torch.Tensor:
    """Return approximately/effectively balanced labels for every batch size."""
    labels = (
        torch.arange(batch_size, dtype=torch.long)
        + int(batch_num)
    ) % 2
    # Avoid a fixed association between batch index and class when batch_size>1.
    if batch_size > 1:
        labels = labels[torch.randperm(batch_size)]
    return labels


def _generate_sequences(
    sequence_length: int,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Generate guaranteed palindromes and guaranteed non-palindromes."""
    batch_size = int(labels.numel())
    half_length = sequence_length // 2

    left_half = torch.randint(
        0,
        2,
        (half_length, batch_size),
        dtype=torch.long,
    )
    right_half = torch.flip(left_half, dims=[0]).clone()

    negative_indices = torch.nonzero(labels.eq(0), as_tuple=False).flatten()
    for batch_index in negative_indices.tolist():
        pair_index = random.randrange(half_length)
        mirrored_index = half_length - 1 - pair_index
        right_half[mirrored_index, batch_index] = (
            1 - right_half[mirrored_index, batch_index]
        )

    return torch.cat([left_half, right_half], dim=0)


def is_even_palindrome(sequence: torch.Tensor) -> torch.Tensor:
    """Check a tensor of shape (length, batch) for palindrome membership."""
    if sequence.dim() != 2:
        raise ValueError("sequence must have shape (length, batch)")
    if sequence.size(0) % 2 != 0:
        return torch.zeros(sequence.size(1), dtype=torch.bool)
    return sequence.eq(torch.flip(sequence, dims=[0])).all(dim=0)


def dataloader(num_batches, batch_size, min_len, max_len):
    """Yield balanced palindrome-classification batches.

    Input shape:
        ``(n + 1, batch, 3)``
        channels 0/1 encode the symbol and channel 2 is the delimiter.

    Target shape:
        ``(n/2, batch, 1)``
        only the final time step contains the binary class target.

    Metadata includes a same-shaped ``loss_mask`` selecting only the final
    classification step. No class labels are exposed to the advice builder.
    """
    for batch_num in range(1, num_batches + 1):
        seq_len = _choose_even_length(min_len, max_len)
        labels = _balanced_labels(batch_num, batch_size)
        symbols = _generate_sequences(seq_len, labels)

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
            "loss_mask": loss_mask,
        }
        yield batch_num, inp, outp, metadata


@attrs
class EvenPalindromeTaskParams(object):
    name = attrib(default="even-palindrome-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)

    # Two read heads allow one mirrored pair to be retrieved in one processing
    # step. In this codebase num_heads creates an equal number of read/write
    # pairs, so this also creates two write heads.
    num_heads = attrib(default=2, converter=int)

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
        )

    @dataloader.default
    def default_dataloader(self):
        return dataloader(
            self.params.num_batches,
            self.params.batch_size,
            self.params.sequence_min_len,
            self.params.sequence_max_len,
        )

    @criterion.default
    def default_criterion(self):
        # The generic training loop applies this only when no mask is present.
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
        return build_even_palindrome_advice(
            metadata["sequence_length"],
            self.params.advice_type,
            random_seed=(
                self.params.advice_random_seed
                + int(metadata["batch_num"])
            ),
        )
