# CartPole-v1 DQN (PyTorch)

A clean, educational Deep Q-Network implementation that solves CartPole-v1.

## Setup

```bash
cd dqn_cartpole

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Project Structure

```
dqn_cartpole/
├── configs/
│   └── dqn_cartpole.yaml       # All hyperparameters
├── outputs/
│   ├── checkpoints/            # best.pt, last.pt
│   ├── logs/                   # training_log.json
│   ├── plots/                  # reward_curve.png
│   └── videos/                 # eval_episode.mp4  (optional)
├── src/
│   ├── utils.py                # Seed, moving average, dir helpers
│   ├── model.py                # QNetwork MLP
│   ├── replay_buffer.py        # Experience replay buffer
│   ├── agent.py                # DQN agent (epsilon-greedy, target net, Huber loss)
│   ├── train.py                # Training loop
│   └── eval.py                 # Evaluation + plots + video
├── requirements.txt
└── README.md
```

## Usage

### 1. Train

```bash
python src/train.py
# or explicitly:
python src/train.py --config configs/dqn_cartpole.yaml
```

Saves `outputs/checkpoints/best.pt` and `last.pt`.  
Logs every episode: reward, 100-ep moving average, epsilon, loss.  
Stops early once the 100-ep mean reward ≥ 475 (solved).

### 2. Evaluate

```bash
# Evaluate best checkpoint (20 episodes, greedy, + plot)
python src/eval.py

# Explicit options
python src/eval.py \
    --config      configs/dqn_cartpole.yaml \
    --checkpoint  outputs/checkpoints/best.pt \
    --episodes    20 \
    --plot            # save reward_curve.png  (default: on)
    --no-plot         # skip plot
    --video           # record one episode to eval_episode.mp4
```

Prints mean ± std reward and a SOLVED / not-solved status line.

### 3. Plot only (no eval)

The `--plot` flag reads `outputs/logs/training_log.json` — you can
regenerate the plot any time without re-running evaluation:

```bash
python src/eval.py --no-plot   # skip (example of turning it off)
python src/eval.py --plot      # regenerate reward_curve.png
```

### 4. Record a video

```bash
python src/eval.py --video
# output: outputs/videos/eval_episode.mp4
```

Requires `imageio` and `imageio-ffmpeg` (both in `requirements.txt`).

## Hyperparameters

Edit `configs/dqn_cartpole.yaml`. Key values:

| Parameter | Default | Notes |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 64 | Mini-batch size |
| `buffer_size` | 50 000 | Replay buffer capacity |
| `epsilon_start/end` | 1.0 → 0.05 | Exploration schedule |
| `epsilon_decay_steps` | 20 000 | Linear decay over N env steps |
| `target_update_freq` | 1 000 | Hard target-network update interval |
| `max_episodes` | 500 | Increase to 1000 to reliably hit solve threshold |

## Expected Results

| Metric | Value |
|---|---|
| Solve threshold | 100-ep mean ≥ 475 |
| Typical solve episode | ~400–700 (seed-dependent) |
| Eval mean (best.pt, 20 ep) | ≥ 475 once solved |
