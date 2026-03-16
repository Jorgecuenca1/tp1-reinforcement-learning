# RL Training Lab - Plataforma de Aprendizaje por Refuerzo

Plataforma web interactiva para entrenar, visualizar y comparar algoritmos de Aprendizaje por Refuerzo.

Desarrollada como Desafio Practico N.1 para la materia **Aprendizaje por Refuerzo I** - Maestria en Inteligencia Artificial - UBA 2026.

---

## Requisitos previos

- **Python 3.10+** (probado con Python 3.13)
- **pip** (gestor de paquetes de Python)
- **Git**

---

## Instalacion paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jorgecuenca1/tp1-reinforcement-learning.git
cd tp1-reinforcement-learning
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv .venv
```

Activar el entorno virtual:

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **Linux / Mac:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para los entornos que usan Box2D (LunarLander):

```bash
pip install swig
pip install "gymnasium[box2d]"
```

### 4. Crear la base de datos

```bash
python manage.py migrate
```

### 5. Cargar las tecnicas y problemas

```bash
python manage.py seed_data
```

Esto carga las 6 tecnicas de RL y los 11 problemas preconfigurados.

### 6. Ejecutar el servidor

```bash
python manage.py runserver
```

Abrir en el navegador: **http://127.0.0.1:8000/**

---

## Uso de la plataforma

### Pagina principal

Desde la pagina de inicio se pueden ver las 6 tecnicas disponibles. Al hacer clic en una tecnica se muestran los problemas compatibles.

### Entrenar un algoritmo

1. Seleccionar una tecnica (ej: Q-Learning)
2. Elegir un problema (ej: Cliff Walking)
3. Configurar los hiperparametros (episodios, gamma, epsilon, etc.)
4. Presionar **Iniciar Entrenamiento**
5. Esperar a que termine (se muestra un spinner de carga)
6. Ver los resultados: grafico de convergencia, metricas y tabla de hiperparametros

### Snake IA

Acceder desde el menu **Snake IA** en la barra de navegacion.

1. Configurar episodios (500-2000 recomendado), tamano del grid y parametros de la red
2. Presionar **Entrenar Snake IA**
3. Una vez terminado, presionar **Reproducir** para ver al agente jugar
4. Usar el slider de velocidad para ajustar la animacion
5. Seleccionar diferentes replays del entrenamiento desde el dropdown

### MARL Predator-Prey

Acceder desde el menu **MARL** en la barra de navegacion.

1. Configurar el grid, numero de depredadores, presas y parametros de entrenamiento
2. Presionar **Entrenar Depredadores**
3. Ver la animacion de los depredadores (rojo) cazando cooperativamente a la presa (azul)
4. Comparar replays de diferentes etapas del entrenamiento

### Exportar a Google Colab

Desde la pagina de resultados de cualquier ejecucion, presionar **Exportar a Colab** para descargar un archivo `.py` autocontenido que se puede ejecutar directamente en Google Colab.

### Comparar ejecuciones

Desde el menu **Comparar**, seleccionar multiples ejecuciones para ver sus curvas de convergencia superpuestas y una tabla comparativa de metricas.

---

## Estructura del proyecto

```
tp1-reinforcement-learning/
├── manage.py                  # Entry point de Django
├── requirements.txt           # Dependencias
├── rl_project/                # Configuracion del proyecto Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── rl_app/                    # Aplicacion principal
│   ├── models.py              # Modelos: Technique, Problem, TrainingRun
│   ├── views.py               # Vistas
│   ├── urls.py                # Rutas
│   ├── admin.py               # Administracion
│   ├── algorithms/            # Implementaciones de los algoritmos
│   │   ├── utils.py           # Utilidades: discretizacion, graficos
│   │   ├── monte_carlo_es.py  # Monte Carlo Exploring Starts
│   │   ├── monte_carlo_is.py  # Monte Carlo Importance Sampling
│   │   ├── sarsa.py           # SARSA
│   │   ├── q_learning.py      # Q-Learning
│   │   ├── dqn.py             # Deep Q-Network (PyTorch)
│   │   ├── snake_env.py       # Entorno custom Snake (Gymnasium)
│   │   ├── snake_dqn.py       # DQN para Snake
│   │   ├── predator_prey_env.py  # Entorno MARL Predator-Prey
│   │   └── predator_prey_dqn.py  # IQL con parameter sharing
│   └── management/commands/
│       └── seed_data.py       # Datos iniciales
├── templates/rl_app/          # Templates HTML
│   ├── base.html              # Layout base
│   ├── home.html              # Pagina principal
│   ├── technique_detail.html  # Detalle de tecnica
│   ├── run_training.html      # Formulario de entrenamiento
│   ├── results.html           # Resultados con graficos
│   ├── results_list.html      # Historial de ejecuciones
│   ├── compare.html           # Comparacion de ejecuciones
│   ├── snake_visual.html      # Visualizacion Snake
│   └── predator_prey.html     # Visualizacion MARL
└── static/                    # Archivos estaticos
```

---

## Tecnicas implementadas

| Tecnica | Tipo | Descripcion |
|---------|------|-------------|
| Monte Carlo ES | Monte Carlo | First-visit con Exploring Starts |
| Monte Carlo IS (Ordinario) | Monte Carlo | Off-policy, estimador no sesgado |
| Monte Carlo IS (Ponderado) | Monte Carlo | Off-policy, baja varianza |
| SARSA | TD | On-policy, caminos seguros |
| Q-Learning | TD | Off-policy, converge a Q* |
| DQN | Deep RL | Red neuronal + Experience Replay + Target Network |

## Problemas disponibles

| Problema | Entorno | Dificultad | Tecnicas |
|----------|---------|------------|----------|
| Blackjack | Blackjack-v1 | Avanzado | MC ES, MC IS |
| FrozenLake 8x8 | FrozenLake-v1 | Experto | MC ES, Q-Learning |
| Cliff Walking | CliffWalking-v1 | Avanzado | MC ES, SARSA, Q-Learning |
| Taxi | Taxi-v3 | Avanzado | MC IS, SARSA, Q-Learning |
| MountainCar | MountainCar-v0 | Experto | SARSA, Q-Learning, DQN |
| Acrobot | Acrobot-v1 | Doctorado | SARSA, Q-Learning, DQN |
| CartPole | CartPole-v1 | Avanzado | Q-Learning, DQN |
| LunarLander | LunarLander-v3 | Doctorado | DQN |
| Snake | Custom | Avanzado | DQN |
| Predator-Prey | Custom MARL | Doctorado | IQL + Parameter Sharing |

---

## Tecnologias utilizadas

- **Backend:** Django 4.2, Python 3.13
- **Algoritmos:** NumPy, Gymnasium 1.2
- **Deep Learning:** PyTorch 2.9
- **Graficos:** Matplotlib, Chart.js 4.4
- **Frontend:** Bootstrap 5.3, HTML5 Canvas
- **Base de datos:** SQLite

---

## Ejecucion en Google Colab

Cada algoritmo puede exportarse como un script `.py` independiente para ejecutar en Google Colab. El archivo incluye:

- Instalacion de dependencias (`!pip install gymnasium torch matplotlib`)
- Implementacion completa del algoritmo
- Configuracion del entorno y los hiperparametros utilizados
- Grafico de convergencia con media movil

---

## Valores recomendados para pruebas rapidas

| Ejercicio | Episodios | Tiempo aproximado |
|-----------|-----------|-------------------|
| Q-Learning + Cliff Walking | 1000 | ~3s |
| SARSA + Taxi | 5000 | ~10s |
| DQN + CartPole | 500 | ~30s |
| Snake IA | 500 | ~30s |
| MARL Predator-Prey | 500 | ~60s |
| MC ES + Blackjack | 50000 | ~15s |
