"""
model.py — Q-Network definition.

A simple two-hidden-layer MLP.  Kept small on purpose: CartPole's
observation space is only 4-dimensional, so a big network would
just overfit and train slowly.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Maps state observations to Q-values for each discrete action.

    Architecture: Linear(obs_dim → 128) → ReLU → Linear(128 → 128) → ReLU
                  → Linear(128 → n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, obs_dim)  →  Q-values: (batch, n_actions)"""
        return self.net(x)
