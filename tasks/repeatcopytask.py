"""Repeat-Copy task for a baseline NTM or an advice-augmented NTM."""

import random

from attr import Factory, attrib, attrs
import numpy as np
import torch
from torch import nn, optim

from ntm.aio import EncapsulatedNTM
from .advice_builders import (
    ADVICE_WIDTH,
    REPEAT_COPY_ADVICE_TYPES,
    build_repeat_copy_advice,
    normalize_advice_type,
    uses_advice,
)


def dataloader(
    num_batches,
    batch_size,
    seq_width,
    seq_min_len,
    seq_max_len,
    repeat_min,
    repeat_max,
):
    """Generate Repeat-Copy examples and non-target metadata."""
    reps_mean = (repeat_max + repeat_min) / 2
    reps_var = (((repeat_max - repeat_min + 1) ** 2) - 1) / 12
    reps_std = np.sqrt(reps_var)

    def rpt_normalize(reps):
        # The original task assumes at least two possible repetition counts.
        if reps_std == 0:
            return 0.0
        return (reps - reps_mean) / reps_std

    for batch_num in range(1, num_batches + 1):
        seq_len = random.randint(seq_min_len, seq_max_len)
        reps = random.randint(repeat_min, repeat_max)

        seq = np.random.binomial(
            1,
            0.5,
            (seq_len, batch_size, seq_width),
        )
        seq = torch.from_numpy(seq)

        inp = torch.zeros(seq_len + 2, batch_size, seq_width + 2)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0
        inp[seq_len + 1, :, seq_width + 1] = rpt_normalize(reps)

        outp = torch.zeros(
            seq_len * reps + 1,
            batch_size,
            seq_width + 1,
        )
        outp[:seq_len * reps, :, :seq_width] = seq.clone().repeat(
            reps,
            1,
            1,
        )
        outp[seq_len * reps, :, seq_width] = 1.0

        metadata = {
            "batch_num": batch_num,
            "sequence_length": seq_len,
            "repetitions": reps,
            "input_steps": seq_len + 2,
            "output_steps": seq_len * reps + 1,
        }
        yield batch_num, inp.float(), outp.float(), metadata


@attrs
class RepeatCopyTaskParams(object):
    name = attrib(default="repeat-copy-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=10, converter=int)
    repeat_min = attrib(default=1, converter=int)
    repeat_max = attrib(default=10, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=250000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)

    advice_type = attrib(default="none", converter=normalize_advice_type)
    advice_size = attrib(default=ADVICE_WIDTH, converter=int)
    advice_strength = attrib(default=1.0, converter=float)
    advice_random_seed = attrib(default=2718, converter=int)


@attrs
class RepeatCopyTaskModelTraining(object):
    params = attrib(default=Factory(RepeatCopyTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        if self.params.advice_type not in REPEAT_COPY_ADVICE_TYPES:
            raise ValueError(
                "Invalid Repeat-Copy advice_type {!r}".format(
                    self.params.advice_type
                )
            )
        advice_size = (
            self.params.advice_size
            if uses_advice(self.params.advice_type)
            else 0
        )
        return EncapsulatedNTM(
            self.params.sequence_width + 2,
            self.params.sequence_width + 1,
            self.params.controller_size,
            self.params.controller_layers,
            self.params.num_heads,
            self.params.memory_n,
            self.params.memory_m,
            advice_size=advice_size,
            advice_strength=self.params.advice_strength,
        )

    @dataloader.default
    def default_dataloader(self):
        return dataloader(
            self.params.num_batches,
            self.params.batch_size,
            self.params.sequence_width,
            self.params.sequence_min_len,
            self.params.sequence_max_len,
            self.params.repeat_min,
            self.params.repeat_max,
        )

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
        return build_repeat_copy_advice(
            metadata["sequence_length"],
            metadata["repetitions"],
            self.params.advice_type,
            random_seed=(
                self.params.advice_random_seed
                + int(metadata["batch_num"])
            ),
        )
