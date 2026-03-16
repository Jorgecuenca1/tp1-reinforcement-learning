"""
DQN especializado para Snake con replay visual.
Entrena el agente y graba episodios para visualizacion en el navegador.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Importar y registrar el entorno
from . import snake_env  # noqa: F401
import gymnasium as gym


class SnakeQNetwork(nn.Module):
    def __init__(self, state_dim=11, action_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train(grid_size=10, num_episodes=2000, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01,
          batch_size=64, target_update=10, replay_buffer_size=50000,
          learning_rate=0.001, hidden_size=256, **kwargs):

    env = gym.make("Snake-v1", grid_size=grid_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = SnakeQNetwork(state_dim, action_dim, hidden_size).to(device)
    target_net = SnakeQNetwork(state_dim, action_dim, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(replay_buffer_size)

    rewards_per_episode = []
    scores_per_episode = []
    best_score = 0
    best_episode_history = []
    saved_replays = []
    save_interval = max(1, num_episodes // 10)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        total_reward = 0
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q.argmax(1).item()

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(obs_next, dtype=np.float32)

            memory.push(state, action, reward, next_state, float(terminated))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                bs, ba, br, bns, bd = memory.sample(batch_size)
                st = torch.FloatTensor(bs).to(device)
                at = torch.LongTensor(ba).to(device)
                rt = torch.FloatTensor(br).to(device)
                nst = torch.FloatTensor(bns).to(device)
                dt = torch.FloatTensor(bd).to(device)

                curr_q = policy_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(nst).max(1)[0]
                    target_q = rt + gamma * next_q * (1 - dt)

                loss = nn.SmoothL1Loss()(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        score = info.get("score", 0)
        rewards_per_episode.append(total_reward)
        scores_per_episode.append(score)

        # Guardar el mejor episodio para replay
        if score >= best_score:
            best_score = score
            best_episode_history = env.unwrapped.get_history()

        # Guardar replays periodicos
        if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
            history = env.unwrapped.get_history()
            if len(history) > 5:
                saved_replays.append({
                    "episode": episode + 1,
                    "frames": history,
                    "score": score,
                    "reward": round(total_reward, 1),
                })

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    # Grabar 3 episodios de evaluacion con la politica final
    eval_episodes = []
    for _ in range(3):
        env = gym.make("Snake-v1", grid_size=grid_size)
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        done = False

        while not done:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax(1).item()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = np.array(obs, dtype=np.float32)

        eval_episodes.append({
            "frames": env.unwrapped.get_history(),
            "score": info.get("score", 0),
        })
        env.close()

    # Elegir el mejor episodio entre evaluacion y entrenamiento
    best_eval = max(eval_episodes, key=lambda x: x["score"])
    if best_eval["score"] >= best_score:
        replay_data = best_eval["frames"]
        replay_score = best_eval["score"]
    else:
        replay_data = best_episode_history
        replay_score = best_score

    metrics = {
        "best_score": best_score,
        "max_score_eval": max(e["score"] for e in eval_episodes),
        "avg_score_last_100": round(float(np.mean(scores_per_episode[-100:])), 2),
        "final_epsilon": round(epsilon, 6),
        "network_params": sum(p.numel() for p in policy_net.parameters()),
        "replay_frames": replay_data,
        "replay_score": replay_score,
        "saved_replays": saved_replays,
        "grid_size": grid_size,
        "scores_history": [int(s) for s in scores_per_episode],
    }

    return rewards_per_episode, metrics
