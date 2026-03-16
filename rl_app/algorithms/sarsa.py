"""
SARSA - On-Policy TD Control
State-Action-Reward-State-Action
Actualiza Q(s,a) usando la accion realmente tomada en el siguiente paso.
"""
import numpy as np
from collections import defaultdict
from .utils import make_env, discretize_state, create_discretization_bins


def train(env_id, env_kwargs=None, num_episodes=5000, gamma=0.99,
          alpha=0.1, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
          requires_discretization=False, discretization_bins=20, **kwargs):

    env = make_env(env_id, env_kwargs)
    n_actions = env.action_space.n

    bins = None
    if requires_discretization:
        bins = create_discretization_bins(env, discretization_bins)

    Q = defaultdict(float)
    rewards_per_episode = []
    steps_per_episode = []

    def get_state(obs):
        if bins is not None:
            return discretize_state(np.array(obs, dtype=np.float64), bins)
        if isinstance(obs, (tuple, list)):
            return tuple(obs)
        if isinstance(obs, np.ndarray):
            return tuple(obs.flatten())
        return (obs,)

    def epsilon_greedy(state, eps):
        if np.random.random() < eps:
            return np.random.randint(n_actions)
        q_vals = [Q[(state, a)] for a in range(n_actions)]
        return int(np.argmax(q_vals))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = get_state(obs)
        action = epsilon_greedy(state, epsilon)

        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not done and not truncated:
            obs_next, reward, done, truncated, _ = env.step(action)
            next_state = get_state(obs_next)
            next_action = epsilon_greedy(next_state, epsilon)

            # SARSA update
            td_target = reward + gamma * Q[(next_state, next_action)] * (not done)
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()

    metrics = {
        "q_table_size": len(Q),
        "avg_steps_last_100": round(np.mean(steps_per_episode[-100:]), 1),
        "final_epsilon": round(epsilon, 6),
    }

    return rewards_per_episode, metrics


def get_colab_code(env_id, env_kwargs, hyperparams, problem_name):
    kwargs_str = repr(env_kwargs) if env_kwargs else "{}"
    return f'''# =============================================================
# SARSA (On-Policy TD Control)
# Problema: {problem_name} ({env_id})
# UBA - MIA - Aprendizaje por Refuerzo I - 2026
# =============================================================
# !pip install gymnasium matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

ENV_ID = "{env_id}"
ENV_KWARGS = {kwargs_str}
NUM_EPISODES = {hyperparams.get("num_episodes", 5000)}
GAMMA = {hyperparams.get("gamma", 0.99)}
ALPHA = {hyperparams.get("alpha", 0.1)}
EPSILON_INITIAL = {hyperparams.get("epsilon", 1.0)}
EPSILON_DECAY = {hyperparams.get("epsilon_decay", 0.9995)}
EPSILON_MIN = {hyperparams.get("epsilon_min", 0.01)}
REQUIRES_DISCRETIZATION = {hyperparams.get("requires_discretization", False)}
DISC_BINS = {hyperparams.get("discretization_bins", 20)}

def create_bins(env, n_bins):
    high = np.where(np.isinf(env.observation_space.high), 10.0, env.observation_space.high)
    low = np.where(np.isinf(env.observation_space.low), -10.0, env.observation_space.low)
    return [np.linspace(low[i], high[i], n_bins + 1)[1:-1] for i in range(env.observation_space.shape[0])]

def discretize(state, bins):
    return tuple(max(0, min(np.digitize(state[i], bins[i]) - 1, len(bins[i]) - 2)) for i in range(len(state)))

def get_state(obs, bins):
    if bins is not None:
        return discretize(np.array(obs, dtype=np.float64), bins)
    if isinstance(obs, (tuple, list)):
        return tuple(obs)
    if isinstance(obs, np.ndarray):
        return tuple(obs.flatten())
    return (obs,)

env = gym.make(ENV_ID, **ENV_KWARGS)
n_actions = env.action_space.n
bins = create_bins(env, DISC_BINS) if REQUIRES_DISCRETIZATION else None

Q = defaultdict(float)
rewards_history = []
epsilon = EPSILON_INITIAL

def eps_greedy(state, eps):
    if np.random.random() < eps:
        return np.random.randint(n_actions)
    return int(np.argmax([Q[(state, a)] for a in range(n_actions)]))

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = get_state(obs, bins)
    action = eps_greedy(state, epsilon)
    total_reward = 0
    done = truncated = False

    while not done and not truncated:
        obs_next, reward, done, truncated, _ = env.step(action)
        next_state = get_state(obs_next, bins)
        next_action = eps_greedy(next_state, epsilon)

        td_target = reward + GAMMA * Q[(next_state, next_action)] * (not done)
        Q[(state, action)] += ALPHA * (td_target - Q[(state, action)])

        state = next_state
        action = next_action
        total_reward += reward

    rewards_history.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if (ep + 1) % 500 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episodio {{ep+1}}/{{NUM_EPISODES}} | Reward promedio: {{avg:.2f}} | Epsilon: {{epsilon:.4f}}")

env.close()

window = 100
moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode="valid")
plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.2, color="steelblue", label="Reward por episodio")
plt.plot(range(window-1, len(rewards_history)), moving_avg, color="orangered", linewidth=2, label=f"Media movil (ventana={{window}})")
plt.xlabel("Episodio", fontsize=13)
plt.ylabel("Recompensa", fontsize=13)
plt.title("SARSA - {problem_name}", fontsize=15, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nResultados finales:")
print(f"  Mejor reward: {{max(rewards_history):.2f}}")
print(f"  Promedio ultimos 100: {{np.mean(rewards_history[-100:]):.2f}}")
print(f"  Tamano Q-table: {{len(Q)}}")
'''
