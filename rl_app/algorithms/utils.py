import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
import os
from pathlib import Path


def make_env(env_id, env_kwargs=None):
    kwargs = env_kwargs or {}
    return gym.make(env_id, **kwargs)


def discretize_state(state, bins):
    discrete = []
    for i, val in enumerate(state):
        idx = np.digitize(val, bins[i]) - 1
        idx = max(0, min(idx, len(bins[i]) - 2))
        discrete.append(idx)
    return tuple(discrete)


def create_discretization_bins(env, n_bins=20):
    high = env.observation_space.high
    low = env.observation_space.low
    high = np.where(np.isinf(high), 10.0, high)
    low = np.where(np.isinf(low), -10.0, low)
    bins = []
    for i in range(env.observation_space.shape[0]):
        bins.append(np.linspace(low[i], high[i], n_bins + 1)[1:-1])
    return bins


def plot_convergence(rewards, window, title, save_path, extra_series=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.15, color='#58a6ff', linewidth=0.5, label='Reward por episodio')

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ma_episodes = np.arange(window, len(rewards) + 1)
        ax.plot(ma_episodes, moving_avg, color='#f78166', linewidth=2.5,
                label=f'Media movil (ventana={window})')

        if len(rewards) >= window * 2:
            std_vals = []
            for i in range(len(moving_avg)):
                start = i
                end = i + window
                std_vals.append(np.std(rewards[start:end]))
            std_vals = np.array(std_vals)
            ax.fill_between(ma_episodes, moving_avg - std_vals, moving_avg + std_vals,
                            alpha=0.15, color='#f78166')

    if extra_series:
        colors = ['#7ee787', '#d2a8ff', '#ffd700']
        for idx, (label, data) in enumerate(extra_series.items()):
            color = colors[idx % len(colors)]
            ax.plot(range(1, len(data) + 1), data, color=color, linewidth=1.5,
                    alpha=0.8, label=label)

    ax.set_xlabel('Episodio', fontsize=13, color='#c9d1d9', fontweight='bold')
    ax.set_ylabel('Recompensa', fontsize=13, color='#c9d1d9', fontweight='bold')
    ax.set_title(title, fontsize=16, color='#f0f6fc', fontweight='bold', pad=15)
    ax.tick_params(colors='#8b949e', labelsize=10)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='#8b949e')
    legend = ax.legend(loc='lower right', fontsize=10, facecolor='#161b22',
                       edgecolor='#30363d', labelcolor='#c9d1d9')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def epsilon_greedy_action(Q, state, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
    return int(np.argmax(q_values))


def greedy_action(Q, state, n_actions):
    q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
    return int(np.argmax(q_values))
