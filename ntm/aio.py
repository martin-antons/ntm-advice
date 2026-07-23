"""Assembly and sequence-state management for an NTM."""

from __future__ import annotations

import torch
from torch import nn

from .advice import AdviceMemoryMatrix, DeterministicAdviceReader
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory
from .ntm import NTM


class EncapsulatedNTM(nn.Module):
    """Construct a baseline NTM or an NTM with structural advice.

    ``num_heads`` remains as a backwards-compatible shorthand. When the more
    explicit counts are omitted, it creates the same number of read and write
    heads. Supplying ``num_read_heads`` and ``num_write_heads`` allows tasks
    such as palindrome recognition to use two simultaneous reads but only one
    write.
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
        *,
        num_read_heads=None,
        num_write_heads=None,
        zero_initialize_advice=True,
    ):
        super().__init__()

        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.controller_size = int(controller_size)
        self.controller_layers = int(controller_layers)
        self.num_heads = int(num_heads)  # legacy/public compatibility field
        self.N = int(N)
        self.M = int(M)
        self.advice_size = int(advice_size)
        self.advice_strength = float(advice_strength)
        self.zero_initialize_advice = bool(zero_initialize_advice)

        self.num_read_heads = int(
            self.num_heads if num_read_heads is None else num_read_heads
        )
        self.num_write_heads = int(
            self.num_heads if num_write_heads is None else num_write_heads
        )

        if self.advice_size < 0:
            raise ValueError("advice_size must be non-negative")
        if self.num_read_heads <= 0:
            raise ValueError("at least one read head is required")
        if self.num_write_heads < 0:
            raise ValueError("num_write_heads must be non-negative")

        memory = NTMMemory(self.N, self.M)

        controller_input_size = (
            self.num_inputs
            + self.M * self.num_read_heads
            + self.advice_size
        )
        controller = LSTMController(
            controller_input_size,
            self.controller_size,
            self.controller_layers,
            zero_input_columns=(
                self.advice_size
                if self.zero_initialize_advice
                else 0
            ),
        )

        # Reads are deliberately placed before writes. The NTM core also
        # enforces this ordering, so every read head observes the same memory
        # state M_{t-1} before any write for time step t is applied.
        heads = nn.ModuleList(
            [
                NTMReadHead(memory, self.controller_size)
                for _ in range(self.num_read_heads)
            ]
            + [
                NTMWriteHead(memory, self.controller_size)
                for _ in range(self.num_write_heads)
            ]
        )

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
        self.record_ntm_trace = False
        self.advice_trace = []
        self.ntm_trace = []

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
        record_ntm_trace=False,
    ):
        """Reset all sequence-local state.

        For an advice-enabled model, omitting both ``advice`` and
        ``advice_schedule`` yields an all-zero advice input and therefore acts
        as the architecture-matched zero-advice control.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = int(batch_size)
        self.memory.reset(self.batch_size)
        self.previous_state = self.ntm.create_new_state(self.batch_size)
        self.time_step = 0

        self.record_advice_trace = bool(record_advice_trace)
        self.record_ntm_trace = bool(record_ntm_trace)
        self.advice_trace = []
        self.ntm_trace = []
        self.ntm.set_record_trace(self.record_ntm_trace)

        if not self.uses_advice:
            if advice is not None or advice_schedule is not None:
                raise ValueError("Advice was provided to a baseline NTM")
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
                    "vector": (
                        x.new_zeros(self.batch_size, self.advice_size)
                        if self.uses_advice
                        else None
                    ),
                }
            else:
                trace = {
                    "step": self.time_step,
                    "indices": advice_result.indices.detach().clone(),
                    "weighting": advice_result.weighting.detach().clone(),
                    "vector": advice_result.vector.detach().clone(),
                }
            self.advice_trace.append(trace)

        if self.record_ntm_trace:
            step_trace = dict(self.ntm.last_step_trace)
            step_trace["step"] = self.time_step
            self.ntm_trace.append(step_trace)

        self.time_step += 1
        return output, self.previous_state

    def calculate_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())
