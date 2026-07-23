"""LSTM controller used by the Neural Turing Machine."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter


class LSTMController(nn.Module):
    """An NTM controller based on an LSTM.

    ``zero_input_columns`` can be used to initialize the final columns of the
    first LSTM input matrix to zero. The advice-augmented NTM uses this for the
    advice-specific controller inputs so that advice initially has no effect,
    while gradients can still learn non-zero advice weights immediately.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_layers: int,
        *,
        zero_input_columns: int = 0,
    ):
        super().__init__()

        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.num_layers = int(num_layers)
        self.zero_input_columns = int(zero_input_columns)

        if self.num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.zero_input_columns < 0:
            raise ValueError("zero_input_columns must be non-negative")
        if self.zero_input_columns > self.num_inputs:
            raise ValueError(
                "zero_input_columns cannot exceed the LSTM input width"
            )

        self.lstm = nn.LSTM(
            input_size=self.num_inputs,
            hidden_size=self.num_outputs,
            num_layers=self.num_layers,
        )

        # Learned initial recurrent state, as in the original codebase.
        self.lstm_h_bias = Parameter(
            torch.randn(self.num_layers, 1, self.num_outputs) * 0.05
        )
        self.lstm_c_bias = Parameter(
            torch.randn(self.num_layers, 1, self.num_outputs) * 0.05
        )

        self.reset_parameters()

    def create_new_state(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self) -> None:
        for parameter in self.lstm.parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 0)
            else:
                stdev = 5 / np.sqrt(self.num_inputs + self.num_outputs)
                nn.init.uniform_(parameter, -stdev, stdev)

        if self.zero_input_columns > 0:
            with torch.no_grad():
                self.lstm.weight_ih_l0[
                    :,
                    -self.zero_input_columns :,
                ].zero_()

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x: torch.Tensor, prev_state):
        if x.dim() != 2:
            raise ValueError(
                "controller input must have shape (batch_size, input_size)"
            )
        x = x.unsqueeze(0)
        output, state = self.lstm(x, prev_state)
        return output.squeeze(0), state
