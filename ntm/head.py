"""NTM read and write heads."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _split_cols(mat, lengths):
    """Split a 2D matrix into variable-length column groups."""
    if mat.size(1) != sum(lengths):
        raise ValueError("lengths must sum to the number of columns")

    boundaries = np.cumsum([0] + lengths)
    return [
        mat[:, start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


class NTMHeadBase(nn.Module):
    """Base class for an NTM read or write head."""

    def __init__(self, memory, controller_size):
        super(NTMHeadBase, self).__init__()
        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def is_read_head(self):
        raise NotImplementedError

    def _new_weighting(self, batch_size):
        # memory.reset(batch_size) is called before the head states are created.
        return self.memory.memory.new_zeros(batch_size, self.N)

    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        return self.memory.address(
            k,
            beta,
            g,
            s,
            gamma,
            w_prev,
        )


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # k, beta, g, s, gamma
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(
            controller_size,
            sum(self.read_lengths),
        )
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return self._new_weighting(batch_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        interface = self.fc_read(embeddings)
        k, beta, g, s, gamma = _split_cols(
            interface,
            self.read_lengths,
        )

        weighting = self._address_memory(
            k,
            beta,
            g,
            s,
            gamma,
            w_prev,
        )
        read_vector = self.memory.read(weighting)
        return read_vector, weighting


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # k, beta, g, s, gamma, erase, add
        self.write_lengths = [
            self.M,
            1,
            1,
            3,
            1,
            self.M,
            self.M,
        ]
        self.fc_write = nn.Linear(
            controller_size,
            sum(self.write_lengths),
        )
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return self._new_weighting(batch_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        interface = self.fc_write(embeddings)
        k, beta, g, s, gamma, erase, add = _split_cols(
            interface,
            self.write_lengths,
        )

        erase = torch.sigmoid(erase)
        weighting = self._address_memory(
            k,
            beta,
            g,
            s,
            gamma,
            w_prev,
        )
        self.memory.write(weighting, erase, add)
        return weighting
