"""
eval.py — Evaluate a saved DQN checkpoint, generate plots, and optionally record a video.

Usage:
    # Evaluate best checkpoint + generate plot
    python src/eval.py

    # Explicit options
    python src/eval.py --config configs/dqn_cartpole.yaml \
                       --checkpoint outputs/checkpoints/best.pt \
                       --episodes 20 \
                       --plot \
                       --video
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import ensure_dir, load_config, moving_average, set_global_seed


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(cfg: dict, checkpoint_path: str, n_episodes: int) -> list[float]:
    """Run N episodes with a greedy policy. Returns per-episode rewards."""
    import torch
    import gymnasium as gym
    from agent import DQNAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(cfg["env_id"])
    env.action_space.seed(cfg["seed"] + 999)   # different seed from training

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg, device)
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0    # fully greedy — no exploration at eval time

    rewards = []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total  = 0.0

        for _ in range(cfg["max_steps_per_episode"]):
            action              = agent.select_action(obs, greedy=True)
            obs, reward, term, trunc, _ = env.step(action)
            total              += reward
            if term or trunc:
                break

        rewards.append(total)
        print(f"  Eval episode {ep:>3d}:  reward = {total:.1f}")

    env.close()
    return rewards


# ── Plotting ──────────────────────────────────────────────────────────

def plot_training(log_path: str, plot_dir: str, ma_window: int = 100) -> str:
    """Read training_log.json and save a reward-curve PNG."""
    import matplotlib.pyplot as plt

    ensure_dir(plot_dir)

    with open(log_path) as f:
        log = json.load(f)

    rewards  = log["episode_rewards"]
    episodes = list(range(1, len(rewards) + 1))
    ma       = moving_average(rewards, window=ma_window)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw episode rewards — thin and transparent so the MA stands out
    ax.plot(episodes, rewards, color="steelblue", alpha=0.35,
            linewidth=0.8, label="Episode reward")

    # Moving average
    ax.plot(episodes, ma, color="darkorange", linewidth=2.0,
            label=f"{ma_window}-ep moving average")

    # Solve threshold reference line
    ax.axhline(475, color="green", linestyle="--", linewidth=1.2,
               label="Solve threshold (475)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training on CartPole-v1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, "reward_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ── Video recording ───────────────────────────────────────────────────

def record_video(cfg: dict, checkpoint_path: str, video_dir: str) -> str | None:
    """Record one greedy episode and save it as an MP4.

    Requires imageio and imageio[ffmpeg] (or imageio-ffmpeg).
    Returns the output path, or None if recording fails.
    """
    try:
        import imageio
        import torch
        import gymnasium as gym
        from agent import DQNAgent
    except ImportError as e:
        print(f"[video] Skipping — missing dependency: {e}")
        return None

    ensure_dir(video_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rgb_array mode gives us raw pixel frames to encode
    env = gym.make(cfg["env_id"], render_mode="rgb_array")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg, device)
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0

    frames = []
    obs, _ = env.reset(seed=cfg["seed"])
    total  = 0.0

    for _ in range(cfg["max_steps_per_episode"]):
        frames.append(env.render())
        action              = agent.select_action(obs, greedy=True)
        obs, reward, term, trunc, _ = env.step(action)
        total              += reward
        if term or trunc:
            break

    # Capture the final frame too
    frames.append(env.render())
    env.close()

    out_path = os.path.join(video_dir, "eval_episode.mp4")
    imageio.mimsave(out_path, frames, fps=30)
    print(f"  Total reward in recorded episode: {total:.1f}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DQN checkpoint")
    parser.add_argument("--config",     default="configs/dqn_cartpole.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt",
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes",   type=int, default=None,
                        help="Override eval_episodes from config")
    parser.add_argument("--plot",       action="store_true", default=True,
                        help="Generate reward curve plot (default: on)")
    parser.add_argument("--no-plot",    dest="plot", action="store_false")
    parser.add_argument("--video",      action="store_true", default=False,
                        help="Record one eval episode as MP4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    n_episodes = args.episodes or cfg["eval_episodes"]

    # ── 1. Evaluate ───────────────────────────────────────────────────
    print(f"\nEvaluating checkpoint: {args.checkpoint}")
    print(f"Episodes: {n_episodes}  |  epsilon = 0.0 (greedy)\n")
    rewards = evaluate(cfg, args.checkpoint, n_episodes)

    mean_r = float(np.mean(rewards))
    std_r  = float(np.std(rewards))
    min_r  = float(np.min(rewards))
    max_r  = float(np.max(rewards))

    print(f"\n{'─'*40}")
    print(f"  Mean reward : {mean_r:.1f} ± {std_r:.1f}")
    print(f"  Min / Max   : {min_r:.1f} / {max_r:.1f}")
    solved = "✅  SOLVED" if mean_r >= 475 else "❌  not solved yet"
    print(f"  Status      : {solved}")
    print(f"{'─'*40}\n")

    # ── 2. Plot ───────────────────────────────────────────────────────
    if args.plot:
        log_path = os.path.join(cfg["log_dir"], "training_log.json")
        if os.path.exists(log_path):
            plot_path = plot_training(log_path, cfg["plot_dir"])
            print(f"Plot saved → {plot_path}")
        else:
            print(f"[plot] No training log found at {log_path}, skipping.")

    # ── 3. Video ──────────────────────────────────────────────────────
    if args.video:
        print("\nRecording one episode...")
        video_path = record_video(cfg, args.checkpoint, cfg["video_dir"])
        if video_path:
            print(f"Video saved → {video_path}")
