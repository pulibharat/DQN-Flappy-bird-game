# Flappy Bird DQN

Deep Q-Network (DQN) agent that learns to play Flappy Bird using PyTorch and Gymnasium.

**Best training result:** 131 pipes in a single episode (~59,000 episodes).

## Requirements

- Python 3.10+
- pip

### Python packages

See [`requirements.txt`](requirements.txt):

| Package | Purpose |
|---------|---------|
| `torch` | DQN model and training |
| `gymnasium` | RL environment API |
| `flappy-bird-gymnasium` | Flappy Bird environment |
| `pyyaml` | Hyperparameters from `parameters.yaml` |

### Optional: GPU (CUDA)

For NVIDIA GPUs, install PyTorch with CUDA from [pytorch.org](https://pytorch.org) instead of the default CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The game simulation still runs mostly on CPU; GPU helps slightly with network updates.

## Setup

```bash
git clone https://github.com/shradha-khapra/RL_DQN.git
cd RL_DQN
pip install -r requirements.txt
```

## Project structure

```
RL_DQN/
├── agent.py              # Training and inference loop
├── dqn.py                # Q-network (MLP)
├── experience_replay.py  # Replay memory
├── parameters.yaml       # Hyperparameters
├── game_flappy_bird.py   # Manual play demo
├── requirements.txt
├── README.md
├── LINKEDIN_POST.md      # Project story / LinkedIn write-up
└── runs/                 # Saved models (gitignored)
    └── flappybirdv0.pt
```

## Usage

### Train

```bash
python agent.py flappybirdv0 --train
```

### Test (render game window)

```bash
python agent.py flappybirdv0
```

### Fresh training run

Delete the old checkpoint before retraining so weights do not carry over:

```powershell
Remove-Item runs\flappybirdv0.pt
python agent.py flappybirdv0 --train
```

## Configuration

Edit `parameters.yaml` under the `flappybirdv0` key:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_lidar` | `false` | `false` = 12-dim state; `true` = 180-dim LIDAR |
| `epsilon_init` | `1` | Starting exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration |
| `epsilon_decay` | `0.9995` | Decay per episode |
| `replay_memory_size` | `100000` | Replay buffer capacity |
| `mini_batch_size` | `32` | Batch size for SGD |
| `network_sync_rate` | `1000` | Steps between target-network sync |
| `alpha` | `0.001` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `log_interval` | `100` | Print progress every N episodes |

## Metrics

- **`score`** — number of pipes passed (same as on-screen game score).
- **`reward`** — cumulative step reward (+0.1 alive, +1 pipe, -1 death, -0.5 top hit).

The best model is saved to `runs/flappybirdv0.pt` when `score` improves.

## Rough training time (no render)

| Episodes | Approx. time (CPU / GPU laptop) |
|----------|----------------------------------|
| 10,000 | ~20–40 min |
| 20,000 | ~40–80 min |
| 50,000 | ~1.5–3 hr |
| 59,000+ | ~2–3+ hr |

First pipe often appears around episodes **1,500–3,000**. Long stretches of `last_score=0` are normal.

## License

See repository owner for license terms.

## Story

See [LINKEDIN_POST.md](https://www.linkedin.com/posts/puli-bharat-58040a310_reinforcementlearning-deeplearning-machinelearning-activity-7461398077728374784-sHyl?utm_source=share&utm_medium=member_android&rcm=ACoAAE8VNrABOjjRoua2Ql1Pxljvi0CuBGElZHQ) for the full learning journey, bugs fixed, and lessons learned.
