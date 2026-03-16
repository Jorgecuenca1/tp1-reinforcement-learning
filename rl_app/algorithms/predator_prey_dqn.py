"""
Multi-Agent DQN para Predator-Prey.
Implementa Independent Q-Learning (IQL) con parameter sharing.

Referencia academica:
- Tan, M. (1993). Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents.
- Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
- Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation for MARL.

En esta implementacion:
- Todos los depredadores comparten la misma red neuronal (parameter sharing).
- Cada depredador recibe su observacion local y selecciona su accion independientemente.
- El reward es compartido (team reward) para fomentar cooperacion.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from . import predator_prey_env  # noqa: F401
import gymnasium as gym


class MARLQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class MARLReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        self.buffer.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (np.array(obs), np.array(acts), np.array(rews, dtype=np.float32),
                np.array(next_obs), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train(grid_size=12, n_predators=3, n_prey=1, vision_range=4,
          num_episodes=1000, max_steps_per_ep=200, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05,
          batch_size=128, target_update=15, replay_buffer_size=100000,
          learning_rate=0.0005, hidden_size=256, **kwargs):

    env = gym.make("PredatorPrey-v1",
                   grid_size=grid_size,
                   n_predators=n_predators,
                   n_prey=n_prey,
                   vision_range=vision_range,
                   max_steps=max_steps_per_ep)

    obs_per_agent = env.unwrapped.obs_per_agent
    n_actions = 5  # 5 acciones por agente

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Red compartida para todos los depredadores (parameter sharing)
    policy_net = MARLQNetwork(obs_per_agent, n_actions, hidden_size).to(device)
    target_net = MARLQNetwork(obs_per_agent, n_actions, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = MARLReplayBuffer(replay_buffer_size)

    rewards_per_episode = []
    captures_per_episode = []
    cooperation_per_episode = []
    steps_to_capture = []
    best_capture_time = max_steps_per_ep
    best_episode_history = []

    # Guardar replays periodicos del entrenamiento
    saved_replays = []
    save_interval = max(1, num_episodes // 10)  # guardar ~10 replays

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        episode_captures = 0
        episode_coop = 0
        done = False

        for step in range(max_steps_per_ep):
            if done:
                break

            actions = []
            for i in range(n_predators):
                agent_obs = obs[i * obs_per_agent:(i + 1) * obs_per_agent]
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    with torch.no_grad():
                        q = policy_net(torch.FloatTensor(agent_obs).unsqueeze(0).to(device))
                        action = q.argmax(1).item()
                actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            memory.push(obs, actions, reward, next_obs, float(terminated))
            obs = next_obs
            total_reward += reward

            episode_captures = info.get("captures", 0)
            episode_coop = info.get("cooperation_events", 0)

            if len(memory) >= batch_size:
                _train_step(policy_net, target_net, optimizer, memory,
                            batch_size, gamma, n_predators, obs_per_agent,
                            n_actions, device)

        rewards_per_episode.append(total_reward)
        captures_per_episode.append(episode_captures)
        cooperation_per_episode.append(episode_coop)

        if episode_captures > 0:
            steps_to_capture.append(info.get("steps", max_steps_per_ep))

        if episode_captures > 0:
            capture_steps = info.get("steps", max_steps_per_ep)
            if capture_steps < best_capture_time:
                best_capture_time = capture_steps
                best_episode_history = env.unwrapped.get_history()

        # Guardar replay periodicamente
        if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
            history = env.unwrapped.get_history()
            if len(history) > 3:
                saved_replays.append({
                    "episode": episode + 1,
                    "frames": history,
                    "captures": episode_captures,
                    "steps": info.get("steps", 0),
                    "reward": round(total_reward, 1),
                })

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    # Evaluacion final: 10 episodios greedy, elegir caceria variada
    eval_episodes = []
    eval_captures = 0

    for _ in range(10):
        env = gym.make("PredatorPrey-v1",
                       grid_size=grid_size,
                       n_predators=n_predators,
                       n_prey=n_prey,
                       vision_range=vision_range,
                       max_steps=max_steps_per_ep)
        obs, _ = env.reset()
        done = False
        for _ in range(max_steps_per_ep):
            if done:
                break
            actions = []
            for i in range(n_predators):
                agent_obs = obs[i * obs_per_agent:(i + 1) * obs_per_agent]
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(agent_obs).unsqueeze(0).to(device))
                    action = q.argmax(1).item()
                actions.append(action)
            obs, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

        captured = info.get("captures", 0) > 0
        ep_steps = info.get("steps", max_steps_per_ep)
        if captured:
            eval_captures += 1

        eval_episodes.append({
            "frames": env.unwrapped.get_history(),
            "captures": info.get("captures", 0),
            "steps": ep_steps,
        })
        env.close()

    # Seleccion de replay: elegir captura con duracion MEDIANA (no la mas larga ni la mas corta)
    successful_evals = [e for e in eval_episodes if e["captures"] > 0 and e["steps"] >= 10]
    if successful_evals:
        # Ordenar por duracion y elegir la mediana
        successful_evals.sort(key=lambda e: e["steps"])
        median_idx = len(successful_evals) // 2
        chosen = successful_evals[median_idx]
        best_replay = chosen["frames"]
        best_capture_time = min(best_capture_time, chosen["steps"])
    elif eval_episodes:
        # Sin capturas exitosas largas: elegir aleatoriamente una
        chosen = random.choice(eval_episodes)
        best_replay = chosen["frames"]

    # Usar el replay de evaluacion (mas variado que el de entrenamiento)
    if best_replay:
        best_episode_history = best_replay
    if not best_episode_history:
        best_episode_history = eval_episodes[-1]["frames"]

    capture_rate_last_100 = sum(1 for c in captures_per_episode[-100:] if c > 0) / min(100, len(captures_per_episode))
    avg_steps_capture = np.mean(steps_to_capture) if steps_to_capture else max_steps_per_ep

    metrics = {
        "capture_rate_total": round(sum(1 for c in captures_per_episode if c > 0) / len(captures_per_episode) * 100, 1),
        "capture_rate_last_100": round(capture_rate_last_100 * 100, 1),
        "eval_capture_rate": round(eval_captures / 10 * 100, 1),
        "avg_steps_to_capture": round(float(avg_steps_capture), 1),
        "best_capture_time": best_capture_time,
        "total_cooperation_events": sum(cooperation_per_episode),
        "final_epsilon": round(epsilon, 6),
        "network_params": sum(p.numel() for p in policy_net.parameters()),
        "replay_frames": best_episode_history,
        "saved_replays": saved_replays,
        "grid_size": grid_size,
        "n_predators": n_predators,
        "n_prey": n_prey,
        "captures_history": captures_per_episode,
        "cooperation_history": cooperation_per_episode,
    }

    return rewards_per_episode, metrics


def _train_step(policy_net, target_net, optimizer, memory,
                batch_size, gamma, n_predators, obs_per_agent,
                n_actions, device):
    """Un paso de entrenamiento con el replay buffer."""
    b_obs, b_acts, b_rews, b_next_obs, b_dones = memory.sample(batch_size)

    # Entrenar para cada agente usando parameter sharing
    total_loss = 0
    for i in range(n_predators):
        # Extraer observacion del agente i
        agent_obs = b_obs[:, i * obs_per_agent:(i + 1) * obs_per_agent]
        agent_next_obs = b_next_obs[:, i * obs_per_agent:(i + 1) * obs_per_agent]
        agent_actions = b_acts[:, i]

        obs_t = torch.FloatTensor(agent_obs).to(device)
        next_obs_t = torch.FloatTensor(agent_next_obs).to(device)
        acts_t = torch.LongTensor(agent_actions).to(device)
        rews_t = torch.FloatTensor(b_rews).to(device)
        dones_t = torch.FloatTensor(b_dones).to(device)

        current_q = policy_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target_net(next_obs_t).max(1)[0]
            target_q = rews_t + gamma * next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(current_q, target_q)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
