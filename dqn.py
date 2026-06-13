import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]
        self.use_lidar = params.get("use_lidar", False)
        self.log_interval = params.get("log_interval", 100)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=self.use_lidar,
            audio_on=False,
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        print(f"obs={num_states} actions={num_actions} use_lidar={self.use_lidar}")
        if is_training:
            print(
                "Tip: score=0 for many episodes is normal. "
                "First pipe often appears around episode 1500-3000."
            )

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0
            total_steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_score = 0
        else:
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=device, weights_only=True)
            )
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)

            episode_reward = 0
            episode_score = 0
            done = False

            while not done and episode_reward < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = state.to(device).unsqueeze(0)
                        action = policy_dqn(s).argmax(dim=1).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_score = info.get("score", episode_score)
                episode_reward += reward
                next_state_t = torch.tensor(next_state, dtype=torch.float32)

                if is_training:
                    memory.append((state, action, next_state_t, reward, done))
                    steps += 1
                    total_steps += 1

                    if len(memory) >= self.mini_batch_size:
                        self.optimize(
                            memory.sample(self.mini_batch_size),
                            policy_dqn,
                            target_dqn,
                        )
                        if steps >= self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0

                state = next_state_t

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_score > best_score:
                    print(
                        f"new best score={episode_score} "
                        f"reward={episode_reward:.1f} episode={episode + 1} "
                        f"epsilon={epsilon:.3f}"
                    )
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_score = episode_score

                if (episode + 1) % self.log_interval == 0:
                    print(
                        f"episode={episode + 1} last_score={episode_score} "
                        f"best_score={best_score} reward={episode_reward:.1f} "
                        f"epsilon={epsilon:.3f}"
                    )
            else:
                print(f"score={episode_score} reward={episode_reward:.1f}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
