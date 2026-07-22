"""Differentiable task memory for an NTM."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _convolve(weighting, shift):
    """Circular convolution for a three-position shift distribution."""
    if shift.size(0) != 3:
        raise ValueError("shift distribution must have size 3")

    padded = torch.cat([
        weighting[-1:],
        weighting,
        weighting[:1],
    ])
    return F.conv1d(
        padded.view(1, 1, -1),
        shift.view(1, 1, -1),
    ).view(-1)


class NTMMemory(nn.Module):
    """Batch of N x M differentiable memory matrices."""

    def __init__(self, N, M):
        super(NTMMemory, self).__init__()
        self.N = N
        self.M = M

        self.register_buffer("mem_bias", torch.empty(N, M))
        stdev = 1 / np.sqrt(N + M)
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

        self.batch_size = None
        self.memory = None

    def reset(self, batch_size):
        self.batch_size = int(batch_size)
        self.memory = self.mem_bias.unsqueeze(0).repeat(
            self.batch_size,
            1,
            1,
        )

    def size(self):
        return self.N, self.M

    def read(self, weighting):
        return torch.matmul(
            weighting.unsqueeze(1),
            self.memory,
        ).squeeze(1)

    def write(self, weighting, erase, add):
        erase_matrix = torch.matmul(
            weighting.unsqueeze(-1),
            erase.unsqueeze(1),
        )
        add_matrix = torch.matmul(
            weighting.unsqueeze(-1),
            add.unsqueeze(1),
        )
        self.memory = (
            self.memory * (1 - erase_matrix)
            + add_matrix
        )

    def address(self, k, beta, g, s, gamma, w_prev):
        content_weighting = self._similarity(k, beta)
        interpolated = self._interpolate(
            w_prev,
            content_weighting,
            g,
        )
        shifted = self._shift(interpolated, s)
        return self._sharpen(shifted, gamma)

    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1)
        similarities = F.cosine_similarity(
            self.memory + 1e-16,
            k + 1e-16,
            dim=-1,
        )
        return F.softmax(beta * similarities, dim=1)

    @staticmethod
    def _interpolate(w_prev, content_weighting, g):
        return g * content_weighting + (1 - g) * w_prev

    def _shift(self, weighting, shift):
        result = torch.zeros_like(weighting)
        for batch_index in range(self.batch_size):
            result[batch_index] = _convolve(
                weighting[batch_index],
                shift[batch_index],
            )
        return result

    @staticmethod
    def _sharpen(weighting, gamma):
        sharpened = weighting ** gamma
        normalizer = sharpened.sum(
            dim=1,
            keepdim=True,
        ) + 1e-16
        return sharpened / normalizer
