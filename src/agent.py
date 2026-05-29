"""
agent.py — DQN agent.

Responsibilities:
  - Epsilon-greedy action selection
  - Maintaining online + target Q-networks
  - Computing TD targets and running one gradient step
  - Hard target-network update
  - Checkpoint save / load
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent for discrete-action environments.

    Args:
        obs_dim:      Dimension of the observation vector.
        n_actions:    Number of discrete actions.
        cfg:          Config dict (loaded from YAML).
        device:       Torch device to run on.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.n_actions = n_actions
        self.device    = device
        self.cfg       = cfg

        # ---- Networks ------------------------------------------------
        self.online_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        # Target starts as an exact copy; weights are frozen (updated manually)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # never in training mode

        # ---- Optimizer + Loss ----------------------------------------
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg["lr"])
        # Huber loss is less sensitive to outlier rewards than MSE
        self.loss_fn = nn.SmoothL1Loss()

        # ---- Replay buffer -------------------------------------------
        self.replay_buffer = ReplayBuffer(
            capacity=cfg["buffer_size"],
            obs_dim=obs_dim,
            device=device,
        )

        # ---- Epsilon schedule ----------------------------------------
        self.epsilon       = cfg["epsilon_start"]
        self._eps_start    = cfg["epsilon_start"]
        self._eps_end      = cfg["epsilon_end"]
        self._eps_decay    = cfg["epsilon_decay_steps"]

        # ---- Step counter (env steps, not gradient steps) ------------
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        Args:
            obs:    Single observation array, shape (obs_dim,).
            greedy: If True, always pick the argmax (used at eval time).
        """
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update_step(self) -> Optional[float]:
        """Sample a mini-batch and run one gradient step.

        Returns the scalar loss value, or None if the buffer isn't
        warm enough yet.
        """
        if len(self.replay_buffer) < self.cfg["min_buffer_size"]:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.cfg["batch_size"]
        )

        # ---- TD target (stop-gradient on target network) -------------
        with torch.no_grad():
            # max Q-value under the target network for the next state
            max_next_q = self.target_net(next_obs).max(dim=1).values
            # If the episode ended (done=1), there is no future return
            td_target = rewards + self.cfg["gamma"] * max_next_q * (1.0 - dones)

        # ---- Online network prediction for taken actions -------------
        # Gather Q(s, a) for the specific actions that were taken
        q_pred = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- Huber loss + gradient step ------------------------------
        loss = self.loss_fn(q_pred, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps stabilise training (common DQN trick)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self) -> None:
        """Linear epsilon decay based on total env steps taken."""
        fraction = min(self.total_steps / self._eps_decay, 1.0)
        self.epsilon = self._eps_start + fraction * (self._eps_end - self._eps_start)

    def update_target_network(self) -> None:
        """Hard copy: copy online network weights into target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save full agent state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore agent state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon     = ckpt.get("epsilon",     self._eps_end)
        self.total_steps = ckpt.get("total_steps", 0)
        self.target_net.eval()
