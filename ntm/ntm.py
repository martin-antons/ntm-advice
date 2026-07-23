#!/usr/bin/env python
"""Core Neural Turing Machine step."""

from __future__ import annotations

import torch
from torch import nn


class NTM(nn.Module):
    """A Neural Turing Machine with optional structural advice input."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        controller,
        memory,
        heads,
        advice_size=0,
    ):
        super().__init__()

        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.controller = controller
        self.memory = memory
        self.heads = heads
        self.advice_size = int(advice_size)

        if self.advice_size < 0:
            raise ValueError("advice_size must be non-negative")

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        self.num_read_heads = 0
        self.num_write_heads = 0
        self.init_r_names = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                name = "read{}_bias".format(self.num_read_heads)
                self.register_buffer(name, init_r_bias)
                self.init_r_names.append(name)
                self.num_read_heads += 1
            else:
                self.num_write_heads += 1

        if self.num_read_heads == 0:
            raise ValueError("heads must contain at least one read head")

        self.fc = nn.Linear(
            self.controller_size + self.num_read_heads * self.M,
            self.num_outputs,
        )
        self.record_trace = False
        self.last_step_trace = {}
        self.reset_parameters()

    def set_record_trace(self, enabled: bool) -> None:
        self.record_trace = bool(enabled)
        self.last_step_trace = {}

    def create_new_state(self, batch_size):
        init_r = [
            getattr(self, name).clone().repeat(batch_size, 1)
            for name in self.init_r_names
        ]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [
            head.create_new_state(batch_size)
            for head in self.heads
        ]
        return init_r, controller_state, heads_state

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(
        self,
        x,
        prev_state,
        advice=None,
        advice_strength=1.0,
    ):
        """Execute one NTM time step.

        All read heads are evaluated before any write head. Thus every read
        vector r_t^(h) is obtained from the same pre-write memory state. Writes
        are then applied sequentially; the supplied tasks use one write head.
        """
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        controller_parts = [x] + prev_reads

        if self.advice_size > 0:
            if advice is None:
                advice = x.new_zeros(x.size(0), self.advice_size)
            if advice.dim() != 2:
                raise ValueError(
                    "advice must have shape (batch_size, advice_size)"
                )
            if advice.size(0) != x.size(0):
                raise ValueError(
                    "Advice batch size {}, expected {}".format(
                        advice.size(0),
                        x.size(0),
                    )
                )
            if advice.size(1) != self.advice_size:
                raise ValueError(
                    "Advice width {}, expected {}".format(
                        advice.size(1),
                        self.advice_size,
                    )
                )

            scaled_advice = advice.to(
                device=x.device,
                dtype=x.dtype,
            ) * float(advice_strength)
            controller_parts.append(scaled_advice)
        elif advice is not None:
            raise ValueError("This NTM was created without an advice input")

        controller_input = torch.cat(controller_parts, dim=1)
        controller_output, controller_state = self.controller(
            controller_input,
            prev_controller_state,
        )

        memory_before = None
        if self.record_trace:
            memory_before = self.memory.memory.detach().clone()

        # Preserve state order even though reads and writes are evaluated in
        # two phases.
        heads_states = [None] * len(self.heads)
        reads = []
        read_weightings = []
        write_weightings = []

        for index, (head, prev_head_state) in enumerate(
            zip(self.heads, prev_heads_states)
        ):
            if not head.is_read_head():
                continue
            read, head_state = head(controller_output, prev_head_state)
            reads.append(read)
            read_weightings.append(head_state)
            heads_states[index] = head_state

        for index, (head, prev_head_state) in enumerate(
            zip(self.heads, prev_heads_states)
        ):
            if head.is_read_head():
                continue
            head_state = head(controller_output, prev_head_state)
            write_weightings.append(head_state)
            heads_states[index] = head_state

        if any(state is None for state in heads_states):
            raise RuntimeError("Not all NTM head states were updated")

        output_input = torch.cat([controller_output] + reads, dim=1)
        output = torch.sigmoid(self.fc(output_input))

        if self.record_trace:
            batch_size = x.size(0)
            self.last_step_trace = {
                "controller_output": controller_output.detach().clone(),
                "read_vectors": torch.stack(reads, dim=1).detach().clone(),
                "read_weightings": torch.stack(
                    read_weightings,
                    dim=1,
                ).detach().clone(),
                "write_weightings": (
                    torch.stack(write_weightings, dim=1).detach().clone()
                    if write_weightings
                    else x.new_zeros(batch_size, 0, self.N)
                ),
                "memory_before": memory_before,
                "memory_after": self.memory.memory.detach().clone(),
                "output": output.detach().clone(),
            }
        else:
            self.last_step_trace = {}

        state = (reads, controller_state, heads_states)
        return output, state
