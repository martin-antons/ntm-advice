"""Assembly and sequence-state management for an NTM."""

import torch
from torch import nn

from .advice import AdviceMemoryMatrix, DeterministicAdviceReader
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory
from .ntm import NTM


class EncapsulatedNTM(nn.Module):
    """Construct a baseline NTM or an NTM with structural advice.

    Existing task code remains compatible because ``advice_size`` defaults to
    zero. Setting ``advice_size > 0`` adds a fixed-width advice vector to the
    controller input.
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        controller_size,
        controller_layers,
        num_heads,
        N,
        M,
        advice_size=0,
        advice_strength=1.0,
    ):
        super(EncapsulatedNTM, self).__init__()

        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.controller_size = int(controller_size)
        self.controller_layers = int(controller_layers)
        self.num_heads = int(num_heads)
        self.N = int(N)
        self.M = int(M)
        self.advice_size = int(advice_size)
        self.advice_strength = float(advice_strength)

        if self.advice_size < 0:
            raise ValueError("advice_size must be non-negative")

        memory = NTMMemory(self.N, self.M)

        controller_input_size = (
            self.num_inputs
            + self.M * self.num_heads
            + self.advice_size
        )
        controller = LSTMController(
            controller_input_size,
            self.controller_size,
            self.controller_layers,
        )

        heads = nn.ModuleList()
        for _ in range(self.num_heads):
            heads.extend([
                NTMReadHead(memory, self.controller_size),
                NTMWriteHead(memory, self.controller_size),
            ])

        self.ntm = NTM(
            self.num_inputs,
            self.num_outputs,
            controller,
            memory,
            heads,
            advice_size=self.advice_size,
        )
        self.memory = memory

        if self.advice_size > 0:
            self.advice_memory = AdviceMemoryMatrix(self.advice_size)
            self.advice_reader = DeterministicAdviceReader(
                self.advice_memory
            )
        else:
            self.advice_memory = None
            self.advice_reader = None

        self.batch_size = None
        self.time_step = 0
        self.record_advice_trace = False
        self.advice_trace = []

    @property
    def uses_advice(self):
        return self.advice_size > 0

    def set_advice_strength(self, value):
        self.advice_strength = float(value)

    def init_sequence(
        self,
        batch_size,
        advice=None,
        advice_schedule=None,
        record_advice_trace=False,
    ):
        """Reset all sequence-local state.

        For an advice-enabled model, ``advice`` and ``advice_schedule`` must be
        supplied together. Omitting both produces an all-zero advice input,
        which is useful as an architectural control.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = int(batch_size)
        self.memory.reset(self.batch_size)
        self.previous_state = self.ntm.create_new_state(
            self.batch_size
        )
        self.time_step = 0

        self.record_advice_trace = bool(record_advice_trace)
        self.advice_trace = []

        if not self.uses_advice:
            if advice is not None or advice_schedule is not None:
                raise ValueError(
                    "Advice was provided to a baseline NTM"
                )
            return

        if (advice is None) != (advice_schedule is None):
            raise ValueError(
                "advice and advice_schedule must be supplied together"
            )

        self.advice_memory.clear()
        self.advice_reader.clear()

        if advice is not None:
            reference = self.memory.memory
            self.advice_memory.load(
                advice,
                self.batch_size,
                device=reference.device,
                dtype=reference.dtype,
            )
            self.advice_reader.load_schedule(
                advice_schedule,
                self.batch_size,
                device=reference.device,
            )

    def forward(self, x=None):
        if self.batch_size is None:
            raise RuntimeError(
                "Call init_sequence(batch_size, ...) before forward()"
            )

        if x is None:
            x = self.memory.memory.new_zeros(
                self.batch_size,
                self.num_inputs,
            )

        advice_read = None
        advice_result = None

        if self.uses_advice and self.advice_reader.is_loaded:
            advice_result = self.advice_reader(self.time_step)
            advice_read = advice_result.vector

        output, self.previous_state = self.ntm(
            x,
            self.previous_state,
            advice=advice_read,
            advice_strength=self.advice_strength,
        )

        if self.record_advice_trace:
            if advice_result is None:
                trace = {
                    "step": self.time_step,
                    "indices": None,
                    "weighting": None,
                    "vector": x.new_zeros(
                        self.batch_size,
                        self.advice_size,
                    ) if self.uses_advice else None,
                }
            else:
                trace = {
                    "step": self.time_step,
                    "indices": advice_result.indices.detach().clone(),
                    "weighting": advice_result.weighting.detach().clone(),
                    "vector": advice_result.vector.detach().clone(),
                }
            self.advice_trace.append(trace)

        self.time_step += 1
        return output, self.previous_state

    def calculate_num_params(self):
        return sum(p.numel() for p in self.parameters())
