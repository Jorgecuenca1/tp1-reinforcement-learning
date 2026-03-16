"""
Deep Q-Network (DQN)
Aproximacion de la funcion Q mediante red neuronal profunda.
Incorpora Experience Replay y Target Network (Mnih et al., 2015).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .utils import make_env


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train(env_id, env_kwargs=None, num_episodes=1000, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
          hidden_layers="128,128", batch_size=64, target_update=10,
          replay_buffer_size=10000, learning_rate_dqn=0.001, **kwargs):

    env = make_env(env_id, env_kwargs)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if isinstance(hidden_layers, str):
        hidden_layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = QNetwork(state_dim, action_dim, hidden_layers).to(device)
    target_net = QNetwork(state_dim, action_dim, hidden_layers).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate_dqn)
    memory = ReplayBuffer(replay_buffer_size)

    rewards_per_episode = []
    losses = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        total_reward = 0
        done = False
        truncated = False
        episode_loss = []

        while not done and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            obs_next, reward, done, truncated, _ = env.step(action)
            next_state = np.array(obs_next, dtype=np.float32)

            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train from replay buffer
            if len(memory) >= batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memory.sample(batch_size)

                states_t = torch.FloatTensor(batch_states).to(device)
                actions_t = torch.LongTensor(batch_actions).to(device)
                rewards_t = torch.FloatTensor(batch_rewards).to(device)
                next_states_t = torch.FloatTensor(batch_next_states).to(device)
                dones_t = torch.FloatTensor(batch_dones).to(device)

                current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                episode_loss.append(loss.item())

        rewards_per_episode.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    metrics = {
        "final_epsilon": round(epsilon, 6),
        "device": str(device),
        "network_params": sum(p.numel() for p in policy_net.parameters()),
        "avg_loss_last_100": round(np.mean(losses[-100:]), 6) if losses else 0,
    }

    return rewards_per_episode, metrics


def get_colab_code(env_id, env_kwargs, hyperparams, problem_name):
    kwargs_str = repr(env_kwargs) if env_kwargs else "{}"
    return f'''# =============================================================
# Deep Q-Network (DQN) con Experience Replay y Target Network
# Problema: {problem_name} ({env_id})
# UBA - MIA - Aprendizaje por Refuerzo I - 2026
# =============================================================
# !pip install gymnasium matplotlib numpy torch

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym

ENV_ID = "{env_id}"
ENV_KWARGS = {kwargs_str}
NUM_EPISODES = {hyperparams.get("num_episodes", 1000)}
GAMMA = {hyperparams.get("gamma", 0.99)}
EPSILON_INITIAL = {hyperparams.get("epsilon", 1.0)}
EPSILON_DECAY = {hyperparams.get("epsilon_decay", 0.9995)}
EPSILON_MIN = {hyperparams.get("epsilon_min", 0.01)}
HIDDEN_LAYERS = {hyperparams.get("hidden_layers", [128, 128])}
BATCH_SIZE = {hyperparams.get("batch_size", 64)}
TARGET_UPDATE = {hyperparams.get("target_update", 10)}
BUFFER_SIZE = {hyperparams.get("replay_buffer_size", 10000)}
LR = {hyperparams.get("learning_rate_dqn", 0.001)}

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d, dtype=np.float32)
    def __len__(self):
        return len(self.buf)

env = gym.make(ENV_ID, **ENV_KWARGS)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {{device}}")

policy_net = QNetwork(state_dim, action_dim, HIDDEN_LAYERS).to(device)
target_net = QNetwork(state_dim, action_dim, HIDDEN_LAYERS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(BUFFER_SIZE)

rewards_history = []
epsilon = EPSILON_INITIAL

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = np.array(obs, dtype=np.float32)
    total_reward = 0
    done = truncated = False

    while not done and not truncated:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax(1).item()

        obs_next, reward, done, truncated, _ = env.step(action)
        next_state = np.array(obs_next, dtype=np.float32)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            bs, ba, br, bns, bd = memory.sample(BATCH_SIZE)
            st = torch.FloatTensor(bs).to(device)
            at = torch.LongTensor(ba).to(device)
            rt = torch.FloatTensor(br).to(device)
            nst = torch.FloatTensor(bns).to(device)
            dt = torch.FloatTensor(bd).to(device)

            curr_q = policy_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(nst).max(1)[0]
                tgt_q = rt + GAMMA * next_q * (1 - dt)

            loss = nn.MSELoss()(curr_q, tgt_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

    rewards_history.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if (ep + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (ep + 1) % 50 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episodio {{ep+1}}/{{NUM_EPISODES}} | Reward promedio: {{avg:.2f}} | Epsilon: {{epsilon:.4f}}")

env.close()

window = min(100, len(rewards_history) // 3) if len(rewards_history) > 30 else 10
moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode="valid")
plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.2, color="steelblue", label="Reward por episodio")
plt.plot(range(window-1, len(rewards_history)), moving_avg, color="orangered", linewidth=2, label=f"Media movil (ventana={{window}})")
plt.xlabel("Episodio", fontsize=13)
plt.ylabel("Recompensa", fontsize=13)
plt.title("DQN - {problem_name}", fontsize=15, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

params = sum(p.numel() for p in policy_net.parameters())
print(f"\\nResultados finales:")
print(f"  Mejor reward: {{max(rewards_history):.2f}}")
print(f"  Promedio ultimos 100: {{np.mean(rewards_history[-100:]):.2f}}")
print(f"  Parametros de la red: {{params}}")
print(f"  Dispositivo: {{device}}")
'''
