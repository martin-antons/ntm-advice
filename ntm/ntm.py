#!/usr/bin/env python
"""Core Neural Turing Machine step."""

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
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads
        self.advice_size = int(advice_size)

        if self.advice_size < 0:
            raise ValueError("advice_size must be non-negative")

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initial previous read values.
        self.num_read_heads = 0
        self.init_r_names = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                name = "read{}_bias".format(self.num_read_heads)
                self.register_buffer(name, init_r_bias)
                self.init_r_names.append(name)
                self.num_read_heads += 1

        if self.num_read_heads == 0:
            raise ValueError("heads must contain at least one read head")

        self.fc = nn.Linear(
            self.controller_size + self.num_read_heads * self.M,
            num_outputs,
        )
        self.reset_parameters()

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

        For an advice-enabled model, the controller input is

            [x_t ; r_(t-1) ; lambda * z_t].

        The advice vector does not directly control a head, the task memory, or
        the output layer. Its interpretation is learned by the controller.
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
            raise ValueError(
                "This NTM was created without an advice input"
            )

        controller_input = torch.cat(controller_parts, dim=1)
        controller_outp, controller_state = self.controller(
            controller_input,
            prev_controller_state,
        )

        reads = []
        heads_states = []
        for head, prev_head_state in zip(
            self.heads,
            prev_heads_states,
        ):
            if head.is_read_head():
                read, head_state = head(
                    controller_outp,
                    prev_head_state,
                )
                reads.append(read)
            else:
                head_state = head(
                    controller_outp,
                    prev_head_state,
                )
            heads_states.append(head_state)

        output_input = torch.cat(
            [controller_outp] + reads,
            dim=1,
        )
        output = torch.sigmoid(self.fc(output_input))

        state = (reads, controller_state, heads_states)
        return output, state
