"""
Snake Environment - Entorno custom compatible con Gymnasium.
La serpiente debe comer comida para crecer. Muere si choca con paredes o consigo misma.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # 4 acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        self.action_space = spaces.Discrete(4)

        # Estado: 11 features binarias/normalizadas
        # [danger_straight, danger_right, danger_left,
        #  dir_up, dir_right, dir_down, dir_left,
        #  food_up, food_right, food_down, food_left]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        self.directions = [
            np.array([-1, 0]),  # arriba
            np.array([0, 1]),   # derecha
            np.array([1, 0]),   # abajo
            np.array([0, -1]),  # izquierda
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        center = self.grid_size // 2
        self.snake = [
            np.array([center, center]),
            np.array([center + 1, center]),
            np.array([center + 2, center]),
        ]
        self.direction_idx = 0  # arriba
        self.food = self._place_food()
        self.steps = 0
        self.steps_since_food = 0
        # Limite de pasos sin comer: da suficiente tiempo para buscar
        self.max_steps_hungry = self.grid_size * self.grid_size * 3
        self.score = 0
        self.done = False

        # Para grabacion visual
        self.history = [self._get_frame_data()]

        return self._get_obs(), {}

    def _place_food(self):
        snake_set = set(tuple(s) for s in self.snake)
        empty = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in snake_set:
                    empty.append((r, c))
        if not empty:
            # Grid lleno: la serpiente gano
            return self.snake[0].copy()
        idx = self.np_random.integers(0, len(empty))
        return np.array(empty[idx])

    def _get_obs(self):
        head = self.snake[0]
        dir_vec = self.directions[self.direction_idx]

        # Direcciones relativas: recto, derecha, izquierda
        dir_right = self.directions[(self.direction_idx + 1) % 4]
        dir_left = self.directions[(self.direction_idx - 1) % 4]

        # Peligro en cada direccion relativa (excluye la cola porque se movera)
        danger_straight = self._is_danger(head + dir_vec)
        danger_right = self._is_danger(head + dir_right)
        danger_left = self._is_danger(head + dir_left)

        # Direccion actual (one-hot)
        dir_up = int(self.direction_idx == 0)
        dir_right_flag = int(self.direction_idx == 1)
        dir_down = int(self.direction_idx == 2)
        dir_left_flag = int(self.direction_idx == 3)

        # Posicion relativa de la comida
        food_up = int(self.food[0] < head[0])
        food_right = int(self.food[1] > head[1])
        food_down = int(self.food[0] > head[0])
        food_left = int(self.food[1] < head[1])

        return np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_right_flag, dir_down, dir_left_flag,
            food_up, food_right, food_down, food_left,
        ], dtype=np.float32)

    def _is_danger(self, point):
        # Fuera del grid
        if point[0] < 0 or point[0] >= self.grid_size:
            return 1
        if point[1] < 0 or point[1] >= self.grid_size:
            return 1
        # Colision con cuerpo EXCLUYENDO la cola (la cola se movera)
        if any(np.array_equal(point, s) for s in self.snake[:-1]):
            return 1
        return 0

    def _is_collision(self, point):
        """Colision real para step: excluye cola (se movera a menos que coma)."""
        if point[0] < 0 or point[0] >= self.grid_size:
            return True
        if point[1] < 0 or point[1] >= self.grid_size:
            return True
        # Excluir la cola: si NO come, la cola se quita.
        # Si come, la cola no se quita, pero la comida esta en new_head
        # no en la cola, asi que no hay conflicto.
        if any(np.array_equal(point, s) for s in self.snake[:-1]):
            return True
        return False

    def step(self, action):
        self.steps += 1
        self.steps_since_food += 1

        # Prevenir giro de 180 grados
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        if action != opposite.get(self.direction_idx, -1):
            self.direction_idx = action

        # Mover cabeza
        new_head = self.snake[0] + self.directions[self.direction_idx]

        # Verificar colision real
        if self._is_collision(new_head):
            self.done = True
            self.history.append(self._get_frame_data())
            return self._get_obs(), -10.0, True, False, {"score": self.score}

        self.snake.insert(0, new_head)

        # Verificar si comio
        reward = -0.01  # penalizacion por paso
        if np.array_equal(new_head, self.food):
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0
            # Verificar si lleno el grid (victoria)
            if len(self.snake) >= self.grid_size * self.grid_size:
                self.done = True
                reward = 100.0
                self.history.append(self._get_frame_data())
                return self._get_obs(), reward, True, False, {"score": self.score}
            self.food = self._place_food()
        else:
            self.snake.pop()  # no crece si no comio

        # Truncar si lleva demasiados pasos sin comer (evita loops infinitos)
        truncated = self.steps_since_food >= self.max_steps_hungry
        if truncated:
            self.done = True

        self.history.append(self._get_frame_data())
        return self._get_obs(), reward, False, truncated, {"score": self.score}

    def _get_frame_data(self):
        """Retorna datos del frame para visualizacion."""
        return {
            "snake": [s.tolist() for s in self.snake],
            "food": self.food.tolist(),
            "score": self.score,
            "direction": self.direction_idx,
        }

    def get_history(self):
        return self.history


# Registrar el entorno en Gymnasium
try:
    gym.spec("Snake-v1")
except (gym.error.NameNotFound, gym.error.NamespaceNotFound):
    gym.register(
        id="Snake-v1",
        entry_point="rl_app.algorithms.snake_env:SnakeEnv",
        kwargs={"grid_size": 10},
    )
