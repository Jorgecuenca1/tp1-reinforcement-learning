"""
Monte Carlo con Exploring Starts (ES)
Metodo on-policy, first-visit Monte Carlo control.
Garantiza exploracion iniciando episodios desde pares (s,a) aleatorios.
"""
import numpy as np
from collections import defaultdict
import gymnasium as gym
from .utils import make_env, discretize_state, create_discretization_bins


def train(env_id, env_kwargs=None, num_episodes=5000, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
          requires_discretization=False, discretization_bins=20, **kwargs):

    env = make_env(env_id, env_kwargs)
    n_actions = env.action_space.n

    bins = None
    if requires_discretization:
        bins = create_discretization_bins(env, discretization_bins)

    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    policy = {}

    rewards_per_episode = []

    def get_state(obs):
        if bins is not None:
            return discretize_state(np.array(obs, dtype=np.float64), bins)
        if isinstance(obs, (tuple, list)):
            return tuple(obs)
        if isinstance(obs, np.ndarray):
            return tuple(obs.flatten())
        return (obs,)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = get_state(obs)

        first_action = np.random.randint(n_actions)

        episode_data = []
        action = first_action
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            obs_next, reward, done, truncated, _ = env.step(action)
            next_state = get_state(obs_next)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

            if not done and not truncated:
                if state in policy:
                    action = policy[state]
                else:
                    action = np.random.randint(n_actions)
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)

        rewards_per_episode.append(total_reward)

        G = 0
        visited = set()
        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[(s, a)] = returns_sum[(s, a)] / returns_count[(s, a)]

                best_a = max(range(n_actions), key=lambda act: Q.get((s, act), 0.0))
                policy[s] = best_a

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()

    metrics = {
        "q_table_size": len(Q),
        "unique_states": len(policy),
        "final_epsilon": round(epsilon, 6),
    }

    return rewards_per_episode, metrics


def get_colab_code(env_id, env_kwargs, hyperparams, problem_name):
    kwargs_str = repr(env_kwargs) if env_kwargs else "{}"
    return f'''# =============================================================
# Monte Carlo con Exploring Starts (ES)
# Problema: {problem_name} ({env_id})
# UBA - MIA - Aprendizaje por Refuerzo I - 2026
# =============================================================
# Para ejecutar en Google Colab:
# !pip install gymnasium matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

# --- Configuracion ---
ENV_ID = "{env_id}"
ENV_KWARGS = {kwargs_str}
NUM_EPISODES = {hyperparams.get("num_episodes", 5000)}
GAMMA = {hyperparams.get("gamma", 0.99)}
EPSILON_INITIAL = {hyperparams.get("epsilon", 1.0)}
EPSILON_DECAY = {hyperparams.get("epsilon_decay", 0.9995)}
EPSILON_MIN = {hyperparams.get("epsilon_min", 0.01)}
REQUIRES_DISCRETIZATION = {hyperparams.get("requires_discretization", False)}
DISC_BINS = {hyperparams.get("discretization_bins", 20)}

# --- Funciones auxiliares ---
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

# --- Entrenamiento ---
env = gym.make(ENV_ID, **ENV_KWARGS)
n_actions = env.action_space.n
bins = create_bins(env, DISC_BINS) if REQUIRES_DISCRETIZATION else None

Q = defaultdict(float)
returns_sum = defaultdict(float)
returns_count = defaultdict(int)
policy = {{}}
rewards_history = []
epsilon = EPSILON_INITIAL

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = get_state(obs, bins)
    action = np.random.randint(n_actions)  # Exploring start

    episode_data = []
    total_reward = 0
    done = truncated = False

    while not done and not truncated:
        obs_next, reward, done, truncated, _ = env.step(action)
        next_state = get_state(obs_next, bins)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state
        if not done and not truncated:
            action = policy.get(state, np.random.randint(n_actions))
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)

    rewards_history.append(total_reward)

    G = 0
    visited = set()
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]
        G = GAMMA * G + r
        if (s, a) not in visited:
            visited.add((s, a))
            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1
            Q[(s, a)] = returns_sum[(s, a)] / returns_count[(s, a)]
            policy[s] = max(range(n_actions), key=lambda act: Q.get((s, act), 0.0))

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if (ep + 1) % 500 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episodio {{ep+1}}/{{NUM_EPISODES}} | Reward promedio (ult. 100): {{avg:.2f}} | Epsilon: {{epsilon:.4f}}")

env.close()

# --- Grafico de convergencia ---
window = 100
moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode="valid")

plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.2, color="steelblue", label="Reward por episodio")
plt.plot(range(window-1, len(rewards_history)), moving_avg, color="orangered", linewidth=2, label=f"Media movil (ventana={{window}})")
plt.xlabel("Episodio", fontsize=13)
plt.ylabel("Recompensa", fontsize=13)
plt.title("Monte Carlo ES - {problem_name}", fontsize=15, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nResultados finales:")
print(f"  Mejor reward: {{max(rewards_history):.2f}}")
print(f"  Promedio ultimos 100: {{np.mean(rewards_history[-100:]):.2f}}")
print(f"  Tamano Q-table: {{len(Q)}} pares (s,a)")
print(f"  Estados unicos: {{len(policy)}}")
'''
