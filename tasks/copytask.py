"""Copy task for a baseline NTM or an advice-augmented NTM."""

from __future__ import annotations

import random

from attr import Factory, attrib, attrs
import numpy as np
import torch
from torch import nn, optim

from ntm.aio import EncapsulatedNTM
from .advice_builders import (
    ADVICE_WIDTH,
    COPY_ADVICE_TYPES,
    build_copy_advice,
    normalize_advice_type,
    uses_advice,
    validate_advice_program,
)


def dataloader(
    num_batches,
    batch_size,
    seq_width,
    min_len,
    max_len,
    *,
    py_rng=None,
    np_rng=None,
):
    """Generate random binary Copy examples and task metadata.

    Optional local RNG objects are used for fixed validation sets without
    consuming or resetting the training RNG streams.
    """
    py_rng = random if py_rng is None else py_rng

    for batch_num in range(1, num_batches + 1):
        seq_len = py_rng.randint(min_len, max_len)
        if np_rng is None:
            seq = np.random.binomial(
                1,
                0.5,
                (seq_len, batch_size, seq_width),
            )
        else:
            seq = np_rng.binomial(
                1,
                0.5,
                (seq_len, batch_size, seq_width),
            )
        seq = torch.from_numpy(seq)

        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0
        outp = seq.clone()

        metadata = {
            "batch_num": batch_num,
            "sequence_length": seq_len,
            "input_steps": seq_len + 1,
            "output_steps": seq_len,
        }
        yield batch_num, inp.float(), outp.float(), metadata


@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)

    advice_type = attrib(default="none", converter=normalize_advice_type)
    advice_size = attrib(default=ADVICE_WIDTH, converter=int)
    advice_strength = attrib(default=1.0, converter=float)
    advice_random_seed = attrib(default=1729, converter=int)


@attrs
class CopyTaskModelTraining(object):
    params = attrib(default=Factory(CopyTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        if self.params.advice_type not in COPY_ADVICE_TYPES:
            raise ValueError(
                "Invalid Copy advice_type {!r}".format(
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
        if (
            self.params.advice_type == "wrong"
            and self.params.sequence_min_len < 2
        ):
            raise ValueError(
                "wrong Copy advice requires sequence_min_len >= 2"
            )

        advice_size = (
            self.params.advice_size
            if uses_advice(self.params.advice_type)
            else 0
        )
        return EncapsulatedNTM(
            self.params.sequence_width + 1,
            self.params.sequence_width,
            self.params.controller_size,
            self.params.controller_layers,
            self.params.num_heads,
            self.params.memory_n,
            self.params.memory_m,
            advice_size=advice_size,
            advice_strength=self.params.advice_strength,
        )

    def make_dataloader(self, *, num_batches=None, seed=None):
        count = (
            self.params.num_batches
            if num_batches is None
            else int(num_batches)
        )
        if seed is None:
            py_rng = None
            np_rng = None
        else:
            py_rng = random.Random(int(seed))
            np_rng = np.random.default_rng(int(seed))

        return dataloader(
            count,
            self.params.batch_size,
            self.params.sequence_width,
            self.params.sequence_min_len,
            self.params.sequence_max_len,
            py_rng=py_rng,
            np_rng=np_rng,
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
        # The random control is deterministic for a task and sequence length.
        # It therefore remains instance- and batch-index-independent.
        program = build_copy_advice(
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
