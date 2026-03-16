"""
Multi-Agent Predator-Prey Environment
Entorno de investigacion MARL (Multi-Agent Reinforcement Learning).
Referencia: Lowe et al. (2017) - MADDPG, OpenAI.

N depredadores deben cooperar para capturar M presas en un grid toroidal.
Los depredadores tienen vision limitada y deben aprender coordinacion emergente.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size=12, n_predators=3, n_prey=1,
                 vision_range=4, max_steps=200, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.vision_range = vision_range
        self._max_steps = max_steps
        self.render_mode = render_mode

        # 5 acciones por agente: quieto, arriba, derecha, abajo, izquierda
        self.action_space = spaces.MultiDiscrete([5] * n_predators)

        # Observacion por depredador:
        # - posicion propia normalizada (2)
        # - distancia relativa a cada presa (2 * n_prey)
        # - distancia relativa a cada otro depredador (2 * (n_predators-1))
        # - si cada presa esta en rango de vision (n_prey)
        obs_size = 2 + 2 * n_prey + 2 * (n_predators - 1) + n_prey
        total_obs = obs_size * n_predators
        self.obs_per_agent = obs_size
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(total_obs,), dtype=np.float32
        )

        self.directions = {
            0: np.array([0, 0]),   # quieto
            1: np.array([-1, 0]),  # arriba
            2: np.array([0, 1]),   # derecha
            3: np.array([1, 0]),   # abajo
            4: np.array([0, -1]), # izquierda
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Colocar agentes aleatoriamente sin superposicion
        positions = set()
        self.predator_pos = []
        for _ in range(self.n_predators):
            while True:
                pos = (self.np_random.integers(0, self.grid_size),
                       self.np_random.integers(0, self.grid_size))
                if pos not in positions:
                    positions.add(pos)
                    self.predator_pos.append(np.array(pos))
                    break

        self.prey_pos = []
        for _ in range(self.n_prey):
            while True:
                pos = (self.np_random.integers(0, self.grid_size),
                       self.np_random.integers(0, self.grid_size))
                if pos not in positions:
                    positions.add(pos)
                    self.prey_pos.append(np.array(pos))
                    break

        self.prey_alive = [True] * self.n_prey
        self.steps = 0
        self.captures = 0
        self.total_captures = 0
        self.cooperation_events = 0

        self.history = [self._get_frame_data()]
        return self._get_obs(), {}

    def _toroidal_dist(self, a, b):
        """Distancia toroidal (el grid se envuelve)."""
        diff = b - a
        diff = np.where(diff > self.grid_size / 2, diff - self.grid_size, diff)
        diff = np.where(diff < -self.grid_size / 2, diff + self.grid_size, diff)
        return diff

    def _manhattan_dist(self, a, b):
        diff = np.abs(self._toroidal_dist(a, b))
        return diff[0] + diff[1]

    def _get_agent_obs(self, agent_idx):
        """Observacion desde la perspectiva de un depredador."""
        pos = self.predator_pos[agent_idx]
        obs = []

        # Posicion propia normalizada
        obs.extend(pos / self.grid_size)

        # Distancia relativa a cada presa
        for i, prey_p in enumerate(self.prey_pos):
            if self.prey_alive[i]:
                rel = self._toroidal_dist(pos, prey_p) / self.grid_size
                obs.extend(rel)
            else:
                obs.extend([0.0, 0.0])

        # Distancia relativa a otros depredadores
        for j, pred_p in enumerate(self.predator_pos):
            if j != agent_idx:
                rel = self._toroidal_dist(pos, pred_p) / self.grid_size
                obs.extend(rel)

        # Presas en rango de vision
        for i, prey_p in enumerate(self.prey_pos):
            if self.prey_alive[i]:
                dist = self._manhattan_dist(pos, prey_p)
                obs.append(1.0 if dist <= self.vision_range else 0.0)
            else:
                obs.append(0.0)

        return np.array(obs, dtype=np.float32)

    def _get_obs(self):
        """Observacion concatenada de todos los depredadores."""
        all_obs = []
        for i in range(self.n_predators):
            all_obs.append(self._get_agent_obs(i))
        return np.concatenate(all_obs)

    def step(self, actions):
        self.steps += 1

        # Mover depredadores (toroidal)
        for i, action in enumerate(actions):
            direction = self.directions[int(action)]
            new_pos = (self.predator_pos[i] + direction) % self.grid_size
            self.predator_pos[i] = new_pos

        # Mover presas (huyen del depredador mas cercano con algo de ruido)
        for i in range(self.n_prey):
            if not self.prey_alive[i]:
                continue
            # Encontrar depredador mas cercano
            min_dist = float('inf')
            closest_pred = 0
            for j in range(self.n_predators):
                d = self._manhattan_dist(self.prey_pos[i], self.predator_pos[j])
                if d < min_dist:
                    min_dist = d
                    closest_pred = j

            # Huir con probabilidad 0.8, movimiento aleatorio con 0.2
            if self.np_random.random() < 0.8 and min_dist <= self.vision_range + 2:
                # Huir: moverse en direccion opuesta al depredador mas cercano
                diff = self._toroidal_dist(self.predator_pos[closest_pred], self.prey_pos[i])
                if abs(diff[0]) >= abs(diff[1]):
                    move = np.array([np.sign(diff[0]), 0], dtype=int)
                else:
                    move = np.array([0, np.sign(diff[1])], dtype=int)
                if move[0] == 0 and move[1] == 0:
                    move = self.directions[self.np_random.integers(1, 5)]
            else:
                move = self.directions[self.np_random.integers(0, 5)]

            self.prey_pos[i] = (self.prey_pos[i] + move) % self.grid_size

        # Verificar capturas (depredador adyacente o en misma celda)
        reward = -0.1  # penalizacion temporal para incentivar capturas rapidas
        capture_this_step = False

        for i in range(self.n_prey):
            if not self.prey_alive[i]:
                continue

            # Contar depredadores cerca de esta presa
            predators_adjacent = []
            for j in range(self.n_predators):
                dist = self._manhattan_dist(self.predator_pos[j], self.prey_pos[i])
                if dist <= 1:  # adyacente o misma celda
                    predators_adjacent.append(j)

            if len(predators_adjacent) >= 2:
                # Captura cooperativa! (necesita 2+ depredadores cerca)
                self.prey_alive[i] = False
                self.captures += 1
                self.total_captures += 1
                self.cooperation_events += 1
                capture_this_step = True
                reward = 50.0  # gran recompensa por captura cooperativa
            elif len(predators_adjacent) == 1:
                # Un solo depredador cerca: recompensa parcial por acercarse
                reward = max(reward, 1.0)

        # Recompensa por acercarse a la presa (reward shaping)
        if not capture_this_step:
            for i in range(self.n_prey):
                if self.prey_alive[i]:
                    min_pred_dist = min(
                        self._manhattan_dist(self.predator_pos[j], self.prey_pos[i])
                        for j in range(self.n_predators)
                    )
                    # Recompensa inversamente proporcional a la distancia
                    proximity_reward = max(0, (self.vision_range - min_pred_dist)) * 0.2
                    reward += proximity_reward

        # Verificar fin
        all_captured = all(not alive for alive in self.prey_alive)
        terminated = all_captured
        truncated = self.steps >= self._max_steps

        if terminated:
            reward += 100.0  # bonus por completar

        self.history.append(self._get_frame_data())
        return self._get_obs(), reward, terminated, truncated, {
            "captures": self.total_captures,
            "cooperation_events": self.cooperation_events,
            "steps": self.steps,
        }

    def _get_frame_data(self):
        return {
            "predators": [p.tolist() for p in self.predator_pos],
            "prey": [p.tolist() for p in self.prey_pos],
            "prey_alive": list(self.prey_alive),
            "captures": self.total_captures,
            "step": self.steps,
        }

    def get_history(self):
        return self.history


# Registrar el entorno
try:
    gym.spec("PredatorPrey-v1")
except (gym.error.NameNotFound, gym.error.NamespaceNotFound):
    gym.register(
        id="PredatorPrey-v1",
        entry_point="rl_app.algorithms.predator_prey_env:PredatorPreyEnv",
        kwargs={"grid_size": 12, "n_predators": 3, "n_prey": 1},
    )
