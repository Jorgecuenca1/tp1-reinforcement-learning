"""
Monte Carlo con Importance Sampling (IS)
Metodo off-policy: politica de comportamiento (epsilon-greedy) y politica objetivo (greedy).
Soporta IS Ordinario e IS Ponderado (Weighted).
"""
import numpy as np
from collections import defaultdict
from .utils import make_env, discretize_state, create_discretization_bins


def train(env_id, env_kwargs=None, num_episodes=5000, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
          variant="weighted", requires_discretization=False,
          discretization_bins=20, **kwargs):

    env = make_env(env_id, env_kwargs)
    n_actions = env.action_space.n

    bins = None
    if requires_discretization:
        bins = create_discretization_bins(env, discretization_bins)

    Q = defaultdict(float)
    C = defaultdict(float)
    target_policy = {}

    rewards_per_episode = []

    def get_state(obs):
        if bins is not None:
            return discretize_state(np.array(obs, dtype=np.float64), bins)
        if isinstance(obs, (tuple, list)):
            return tuple(obs)
        if isinstance(obs, np.ndarray):
            return tuple(obs.flatten())
        return (obs,)

    def behavior_policy_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return target_policy.get(state, np.random.randint(n_actions))

    def behavior_policy_prob(state, action):
        greedy_a = target_policy.get(state, 0)
        if action == greedy_a:
            return (1 - epsilon) + epsilon / n_actions
        return epsilon / n_actions

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = get_state(obs)

        episode_data = []
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = behavior_policy_action(state)
            obs_next, reward, done, truncated, _ = env.step(action)
            next_state = get_state(obs_next)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

        G = 0.0
        W = 1.0

        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r

            if variant == "weighted":
                C[(s, a)] += W
                Q[(s, a)] += (W / C[(s, a)]) * (G - Q[(s, a)])
            else:
                C[(s, a)] += 1
                Q[(s, a)] += (1.0 / C[(s, a)]) * (G - Q[(s, a)])

            best_a = max(range(n_actions), key=lambda act: Q.get((s, act), 0.0))
            target_policy[s] = best_a

            if a != best_a:
                break

            b_prob = behavior_policy_prob(s, a)
            if b_prob < 1e-10:
                break
            W *= 1.0 / b_prob

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()

    metrics = {
        "q_table_size": len(Q),
        "unique_states": len(target_policy),
        "variant": variant,
        "final_epsilon": round(epsilon, 6),
    }

    return rewards_per_episode, metrics


def get_colab_code(env_id, env_kwargs, hyperparams, problem_name):
    kwargs_str = repr(env_kwargs) if env_kwargs else "{}"
    variant = hyperparams.get("variant", "weighted")
    variant_label = "Ponderado" if variant == "weighted" else "Ordinario"
    return f'''# =============================================================
# Monte Carlo Importance Sampling ({variant_label})
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
EPSILON_INITIAL = {hyperparams.get("epsilon", 1.0)}
EPSILON_DECAY = {hyperparams.get("epsilon_decay", 0.9995)}
EPSILON_MIN = {hyperparams.get("epsilon_min", 0.01)}
VARIANT = "{variant}"
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
C = defaultdict(float)
target_policy = {{}}
rewards_history = []
epsilon = EPSILON_INITIAL

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = get_state(obs, bins)
    episode_data = []
    total_reward = 0
    done = truncated = False

    while not done and not truncated:
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = target_policy.get(state, np.random.randint(n_actions))
        obs_next, reward, done, truncated, _ = env.step(action)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = get_state(obs_next, bins)

    rewards_history.append(total_reward)
    G = 0.0
    W = 1.0

    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]
        G = GAMMA * G + r
        if VARIANT == "weighted":
            C[(s, a)] += W
            Q[(s, a)] += (W / C[(s, a)]) * (G - Q[(s, a)])
        else:
            C[(s, a)] += 1
            Q[(s, a)] += (1.0 / C[(s, a)]) * (G - Q[(s, a)])

        best_a = max(range(n_actions), key=lambda act: Q.get((s, act), 0.0))
        target_policy[s] = best_a
        if a != best_a:
            break
        greedy_a = target_policy.get(s, 0)
        b_prob = ((1 - epsilon) + epsilon / n_actions) if a == greedy_a else (epsilon / n_actions)
        if b_prob < 1e-10:
            break
        W *= 1.0 / b_prob

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
plt.title("Monte Carlo IS ({variant_label}) - {problem_name}", fontsize=15, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nResultados finales:")
print(f"  Mejor reward: {{max(rewards_history):.2f}}")
print(f"  Promedio ultimos 100: {{np.mean(rewards_history[-100:]):.2f}}")
print(f"  Tamano Q-table: {{len(Q)}}")
print(f"  Variante: {variant_label}")
'''
