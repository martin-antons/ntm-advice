"""Read-only advice memory and deterministic advice access.

* ``AdviceMemoryMatrix`` stores externally constructed structural prior
  knowledge. It has no write operation and no trainable parameters.
* ``DeterministicAdviceReader`` selects one row at each model time step from
  an externally supplied schedule. It has no trainable parameters.
"""

from typing import NamedTuple, Optional

import torch
from torch import nn


class AdviceReadResult(NamedTuple):
    """Result of one deterministic advice read."""

    vector: torch.Tensor
    weighting: torch.Tensor
    indices: torch.Tensor


class AdviceMemoryMatrix(nn.Module):
    """Sequence-local, read-only advice memory.

    The stored tensor has shape ``(batch_size, num_rows, advice_size)``.
    Advice can be loaded either as a shared matrix of shape
    ``(num_rows, advice_size)`` or as a batch-specific tensor of shape
    ``(batch_size, num_rows, advice_size)``.

    The loaded advice is detached from autograd and is not included in the
    module state dict. It is external information, not a model parameter.
    """

    def __init__(self, advice_size: int):
        super().__init__()
        if advice_size <= 0:
            raise ValueError("advice_size must be positive")

        self.advice_size = int(advice_size)
        self.register_buffer(
            "_memory",
            torch.empty(0),
            persistent=False,
        )

    @property
    def is_loaded(self) -> bool:
        return self._memory.numel() > 0

    @property
    def batch_size(self) -> int:
        self._require_loaded()
        return int(self._memory.size(0))

    @property
    def num_rows(self) -> int:
        self._require_loaded()
        return int(self._memory.size(1))

    @property
    def memory(self) -> torch.Tensor:
        """Return the loaded read-only tensor for inspection."""
        self._require_loaded()
        return self._memory

    def clear(self) -> None:
        """Remove the sequence-local advice."""
        self._memory = self._memory.new_empty(0)

    def load(
        self,
        advice: torch.Tensor,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Load advice for one sequence or batch.

        Args:
            advice:
                Shape ``(num_rows, advice_size)`` for advice shared by the
                batch, or ``(batch_size, num_rows, advice_size)``.
            batch_size:
                Number of sequences in the current batch.
            device:
                Target device. When omitted, the advice tensor's device is used.
            dtype:
                Target floating-point dtype. When omitted, the advice tensor's
                dtype is used.
        """
        if not torch.is_tensor(advice):
            raise TypeError("advice must be a torch.Tensor")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if advice.dim() not in (2, 3):
            raise ValueError(
                "advice must have shape (num_rows, advice_size) or "
                "(batch_size, num_rows, advice_size)"
            )
        if advice.size(-1) != self.advice_size:
            raise ValueError(
                "Expected advice width {}, got {}".format(
                    self.advice_size,
                    advice.size(-1),
                )
            )
        if advice.size(-2) <= 0:
            raise ValueError("advice must contain at least one row")
        if advice.dim() == 3 and advice.size(0) != batch_size:
            raise ValueError(
                "Batch-specific advice has batch size {}, expected {}".format(
                    advice.size(0),
                    batch_size,
                )
            )
        if not advice.is_floating_point():
            advice = advice.float()

        target_device = advice.device if device is None else device
        target_dtype = advice.dtype if dtype is None else dtype

        # Advice is fixed external information. Clone it so later changes to the
        # builder tensor cannot mutate the loaded sequence state.
        loaded = advice.detach().to(
            device=target_device,
            dtype=target_dtype,
        ).clone()

        if loaded.dim() == 2:
            loaded = loaded.unsqueeze(0).expand(batch_size, -1, -1).clone()

        self._memory = loaded

    def read_rows(self, indices: torch.Tensor) -> AdviceReadResult:
        """Read one deterministic row per batch item.

        ``indices`` has shape ``(batch_size,)``. Index ``-1`` means that no
        advice is supplied in this time step; the returned vector and weighting
        are then zero for that batch item.
        """
        self._require_loaded()

        if not torch.is_tensor(indices):
            raise TypeError("indices must be a torch.Tensor")
        if indices.dim() != 1 or indices.size(0) != self.batch_size:
            raise ValueError(
                "indices must have shape ({},), got {}".format(
                    self.batch_size,
                    tuple(indices.size()),
                )
            )

        indices = indices.to(device=self._memory.device, dtype=torch.long)
        active = indices.ge(0)

        if torch.any(indices < -1):
            raise IndexError("Advice indices must be -1 or non-negative")
        if torch.any(indices[active] >= self.num_rows):
            bad = indices[active][indices[active] >= self.num_rows][0].item()
            raise IndexError(
                "Advice row {} is out of range for {} rows".format(
                    bad,
                    self.num_rows,
                )
            )

        safe_indices = indices.clamp_min(0)
        batch_indices = torch.arange(
            self.batch_size,
            device=self._memory.device,
        )
        vectors = self._memory[batch_indices, safe_indices]
        vectors = vectors * active.to(vectors.dtype).unsqueeze(1)

        weightings = self._memory.new_zeros(self.batch_size, self.num_rows)
        if active.any():
            weightings[active, safe_indices[active]] = 1.0

        return AdviceReadResult(
            vector=vectors,
            weighting=weightings,
            indices=indices,
        )

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("No advice matrix is loaded")


class DeterministicAdviceReader(nn.Module):
    """Non-trainable reader controlled by an external row schedule.

    A schedule has shape ``(total_steps,)`` when shared by a batch, or
    ``(total_steps, batch_size)`` for batch-specific access. Each entry is a row
    index in the associated ``AdviceMemoryMatrix``. ``-1`` denotes a time step
    without advice.

    The reader does not interpret the advice and does not learn where to look.
    It only implements the fixed map

        z_t = A[rho(t)]

    where ``rho`` is the supplied schedule.
    """

    def __init__(self, memory: AdviceMemoryMatrix):
        super().__init__()
        self.memory = memory
        self.register_buffer(
            "_schedule",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    @property
    def is_loaded(self) -> bool:
        return self._schedule.numel() > 0

    @property
    def total_steps(self) -> int:
        self._require_loaded()
        return int(self._schedule.size(0))

    @property
    def schedule(self) -> torch.Tensor:
        self._require_loaded()
        return self._schedule

    def clear(self) -> None:
        self._schedule = self._schedule.new_empty((0, 0))

    def load_schedule(
        self,
        schedule: torch.Tensor,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load a deterministic row schedule for one sequence or batch."""
        if not torch.is_tensor(schedule):
            raise TypeError("schedule must be a torch.Tensor")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if schedule.dim() not in (1, 2):
            raise ValueError(
                "schedule must have shape (total_steps,) or "
                "(total_steps, batch_size)"
            )
        if schedule.size(0) <= 0:
            raise ValueError("schedule must contain at least one time step")
        if schedule.dim() == 2 and schedule.size(1) != batch_size:
            raise ValueError(
                "Batch-specific schedule has batch size {}, expected {}".format(
                    schedule.size(1),
                    batch_size,
                )
            )

        target_device = schedule.device if device is None else device
        loaded = schedule.detach().to(
            device=target_device,
            dtype=torch.long,
        ).clone()

        if loaded.dim() == 1:
            loaded = loaded.unsqueeze(1).expand(-1, batch_size).clone()

        self._schedule = loaded

    def forward(self, step: int) -> AdviceReadResult:
        self._require_loaded()

        if not isinstance(step, int):
            raise TypeError("step must be an int")
        if step < 0 or step >= self.total_steps:
            raise IndexError(
                "Advice step {} is outside the loaded schedule [0, {})".format(
                    step,
                    self.total_steps,
                )
            )

        return self.memory.read_rows(self._schedule[step])

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("No deterministic advice schedule is loaded")
