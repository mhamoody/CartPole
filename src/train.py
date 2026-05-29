"""
train.py — DQN training loop for CartPole-v1.

Usage:
    python src/train.py
    python src/train.py --config configs/dqn_cartpole.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Allow running from repo root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import DQNAgent
from src.utils import ensure_dir, load_config, moving_average, set_global_seed


# ── CartPole is considered "solved" when the 100-episode mean ≥ 475 ──
SOLVE_THRESHOLD = 475.0


def make_env(cfg: dict):
    """Create and return a seeded CartPole-v1 environment."""
    import gymnasium as gym

    env = gym.make(cfg["env_id"])
    # Seed the action space for reproducible random actions during warmup
    env.action_space.seed(cfg["seed"])
    return env


def warmup_replay(env, agent: DQNAgent, min_size: int) -> None:
    """Fill the replay buffer with random transitions before training.

    This ensures the first mini-batch isn't drawn from a near-empty buffer,
    which would cause highly correlated, low-diversity updates.
    """
    print(f"Warming up replay buffer to {min_size} transitions...")
    obs, _ = env.reset()

    while len(agent.replay_buffer) < min_size:
        action = env.action_space.sample()          # pure random policy
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.add(obs, action, reward, next_obs, done)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    print(f"Replay buffer ready: {len(agent.replay_buffer)} transitions.\n")


def train(cfg: dict) -> None:
    # ── Setup ─────────────────────────────────────────────────────────
    set_global_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])

    env = make_env(cfg)

    obs_dim  = env.observation_space.shape[0]   # 4 for CartPole-v1
    n_actions = env.action_space.n              # 2 for CartPole-v1

    agent = DQNAgent(obs_dim, n_actions, cfg, device)

    # ── Warmup ────────────────────────────────────────────────────────
    warmup_replay(env, agent, cfg["min_buffer_size"])

    # ── Training state ────────────────────────────────────────────────
    episode_rewards: list[float] = []
    episode_losses:  list[float] = []   # mean loss per episode
    best_mean_reward = -np.inf
    t_start = time.time()

    print(f"{'Episode':>8}  {'Reward':>8}  {'MA-100':>8}  {'Epsilon':>8}  {'Loss':>10}  {'Steps':>8}")
    print("-" * 68)

    # ── Main loop ─────────────────────────────────────────────────────
    for episode in range(1, cfg["max_episodes"] + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        losses_this_ep: list[float] = []

        for _ in range(cfg["max_steps_per_episode"]):
            # 1. Select action (epsilon-greedy)
            action = agent.select_action(obs)

            # 2. Step the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. Store transition
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            agent.total_steps += 1

            # 4. Decay epsilon (linear, based on total env steps)
            agent.update_epsilon()

            # 5. Optimise online network every `train_freq` steps
            if agent.total_steps % cfg["train_freq"] == 0:
                loss = agent.update_step()
                if loss is not None:
                    losses_this_ep.append(loss)

            # 6. Hard-update target network every `target_update_freq` steps
            if agent.total_steps % cfg["target_update_freq"] == 0:
                agent.update_target_network()

            if done:
                break

        # ── End-of-episode bookkeeping ────────────────────────────────
        episode_rewards.append(episode_reward)
        mean_loss = float(np.mean(losses_this_ep)) if losses_this_ep else 0.0
        episode_losses.append(mean_loss)

        # Moving average over last 100 episodes (or however many exist)
        window = min(100, len(episode_rewards))
        mean_100 = float(np.mean(episode_rewards[-window:]))

        # ── Logging ───────────────────────────────────────────────────
        print(
            f"{episode:>8d}  {episode_reward:>8.1f}  {mean_100:>8.2f}"
            f"  {agent.epsilon:>8.4f}  {mean_loss:>10.6f}  {agent.total_steps:>8d}"
        )

        # ── Checkpointing ─────────────────────────────────────────────
        # Always save last.pt so training can be inspected / resumed
        last_path = os.path.join(cfg["checkpoint_dir"], "last.pt")
        agent.save_checkpoint(last_path)

        # Save best.pt when the 100-episode mean improves
        if mean_100 > best_mean_reward:
            best_mean_reward = mean_100
            best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
            agent.save_checkpoint(best_path)

        # ── Early stopping ────────────────────────────────────────────
        # Only check once we have 100 episodes of data
        if len(episode_rewards) >= 100 and mean_100 >= SOLVE_THRESHOLD:
            elapsed = time.time() - t_start
            print(f"\n🎉  Solved at episode {episode}! "
                  f"100-ep mean = {mean_100:.1f}  ({elapsed:.0f}s)")
            break

    # ── Save training log (JSON) ───────────────────────────────────────
    log = {
        "episode_rewards": episode_rewards,
        "episode_losses":  episode_losses,
        "best_mean_reward": best_mean_reward,
        "total_steps":     agent.total_steps,
        "config":          cfg,
    }
    log_path = os.path.join(cfg["log_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved → {log_path}")

    env.close()


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument(
        "--config",
        default="configs/dqn_cartpole.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
