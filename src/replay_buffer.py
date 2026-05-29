"""
replay_buffer.py — Fixed-size circular experience replay buffer.

Stores (s, a, r, s', done) tuples and returns random mini-batches
as PyTorch tensors ready for the update step.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer that stores transitions and samples mini-batches.

    Args:
        capacity:   Maximum number of transitions to keep.
        obs_dim:    Dimensionality of a single observation.
        device:     Torch device tensors are moved to on sampling.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._ptr = 0       # next write position
        self._size = 0      # current number of stored transitions

        # Pre-allocate numpy arrays — faster than a list of tuples
        self._obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros((capacity,),         dtype=np.int64)
        self._rewards  = np.zeros((capacity,),         dtype=np.float32)
        self._dones    = np.zeros((capacity,),         dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest if full."""
        self._obs[self._ptr]      = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._dones[self._ptr]    = float(done)

        # Advance pointer circularly
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Return a random mini-batch as (obs, actions, rewards, next_obs, dones).

        All tensors are on self.device.
        dones is float32 (1.0 = terminal) for easy use in the TD target.
        """
        idxs = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.tensor(self._obs[idxs],      device=self.device),
            torch.tensor(self._actions[idxs],  device=self.device),
            torch.tensor(self._rewards[idxs],  device=self.device),
            torch.tensor(self._next_obs[idxs], device=self.device),
            torch.tensor(self._dones[idxs],    device=self.device),
        )

    def __len__(self) -> int:
        return self._size
