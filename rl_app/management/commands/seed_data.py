from django.core.management.base import BaseCommand
from rl_app.models import Technique, Problem


TECHNIQUES = [
    {
        "name": "Monte Carlo ES",
        "slug": "monte-carlo-es",
        "description": """Monte Carlo con Exploring Starts (ES) es un metodo on-policy de control que garantiza la exploracion completa del espacio estado-accion iniciando cada episodio desde un par (estado, accion) seleccionado aleatoriamente.

Fundamento teorico: Se basa en el teorema de convergencia de Monte Carlo first-visit, donde Q(s,a) converge al valor esperado del retorno G_t cuando el numero de visitas tiende a infinito. La condicion de Exploring Starts asegura que todos los pares (s,a) son visitados infinitas veces en el limite.

Ecuacion de actualizacion:
Q(s,a) <- promedio(Returns(s,a))

donde Returns(s,a) acumula los retornos G_t observados desde la primera visita al par (s,a) en cada episodio.""",
        "category": "monte_carlo",
        "algorithm_key": "monte_carlo_es",
        "icon": "dice-5",
        "order": 1,
        "pseudocode": """Inicializar Q(s,a) arbitrariamente, pi(s) arbitrariamente
Inicializar Returns(s,a) <- lista vacia para todo s,a

Repetir para cada episodio:
  Elegir S0, A0 aleatoriamente (Exploring Starts)
  Generar episodio siguiendo pi: S0,A0,R1,...,ST
  G <- 0
  Para t = T-1, T-2, ..., 0:
    G <- gamma * G + R_{t+1}
    Si (St, At) no aparece en S0,A0,...,S_{t-1},A_{t-1}:
      Agregar G a Returns(St, At)
      Q(St, At) <- promedio(Returns(St, At))
      pi(St) <- argmax_a Q(St, a)""",
    },
    {
        "name": "Monte Carlo IS (Ordinario)",
        "slug": "monte-carlo-is-ordinario",
        "description": """Monte Carlo con Importance Sampling Ordinario es un metodo off-policy que utiliza una politica de comportamiento (b) epsilon-greedy para explorar, mientras estima el valor de una politica objetivo (pi) greedy.

El ratio de importance sampling corrige el sesgo introducido por muestrear trayectorias de b en lugar de pi:

rho_{t:T-1} = prod_{k=t}^{T-1} pi(A_k|S_k) / b(A_k|S_k)

IS Ordinario: V(s) = sum(rho_t * G_t) / N

Este estimador es no sesgado pero puede tener alta varianza, especialmente cuando los ratios de importance sampling son grandes.""",
        "category": "monte_carlo",
        "algorithm_key": "monte_carlo_is_ordinary",
        "icon": "arrow-left-right",
        "order": 2,
        "pseudocode": """Inicializar Q(s,a), C(s,a) <- 0
Politica objetivo pi: greedy respecto a Q

Repetir para cada episodio:
  b <- politica epsilon-greedy
  Generar episodio con b: S0,A0,R1,...,ST
  G <- 0, W <- 1
  Para t = T-1, ..., 0:
    G <- gamma * G + R_{t+1}
    C(St,At) <- C(St,At) + 1
    Q(St,At) <- Q(St,At) + (1/C) * (G - Q(St,At))
    pi(St) <- argmax_a Q(St,a)
    Si At != pi(St): break
    W <- W * (1 / b(At|St))""",
    },
    {
        "name": "Monte Carlo IS (Ponderado)",
        "slug": "monte-carlo-is-ponderado",
        "description": """Monte Carlo con Importance Sampling Ponderado (Weighted IS) es la variante preferida del IS off-policy. A diferencia del IS Ordinario, el Weighted IS produce un estimador sesgado pero con varianza significativamente menor.

IS Ponderado: V(s) = sum(rho_t * G_t) / sum(rho_t)

La normalizacion por la suma de los pesos reduce dramaticamente la varianza. El sesgo desaparece asintoticamente (es consistente). En la practica, converge mucho mas rapido que el IS Ordinario.

Referencia: Sutton & Barto (2018), Capitulo 5.6-5.7.""",
        "category": "monte_carlo",
        "algorithm_key": "monte_carlo_is_weighted",
        "icon": "balance-scale",
        "order": 3,
        "pseudocode": """Inicializar Q(s,a), C(s,a) <- 0
Politica objetivo pi: greedy respecto a Q

Repetir para cada episodio:
  b <- politica epsilon-greedy
  Generar episodio con b: S0,A0,R1,...,ST
  G <- 0, W <- 1
  Para t = T-1, ..., 0:
    G <- gamma * G + R_{t+1}
    C(St,At) <- C(St,At) + W
    Q(St,At) <- Q(St,At) + (W/C) * (G - Q(St,At))
    pi(St) <- argmax_a Q(St,a)
    Si At != pi(St): break
    W <- W * (1 / b(At|St))""",
    },
    {
        "name": "SARSA",
        "slug": "sarsa",
        "description": """SARSA (State-Action-Reward-State-Action) es un algoritmo de control on-policy basado en Temporal Difference (TD). Actualiza Q(s,a) utilizando la accion efectivamente tomada en el siguiente paso bajo la misma politica.

Ecuacion de actualizacion:
Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Al ser on-policy, SARSA tiene en cuenta el riesgo de la politica de exploracion. Esto lo hace mas conservador que Q-Learning: en entornos con penalizaciones severas (como CliffWalking), SARSA aprende caminos mas seguros.

Convergencia: Converge a la politica optima epsilon-greedy bajo las condiciones de Robbins-Monro para la tasa de aprendizaje.""",
        "category": "td",
        "algorithm_key": "sarsa",
        "icon": "arrow-repeat",
        "order": 4,
        "pseudocode": """Inicializar Q(s,a) para todo s,a (ej: 0)

Repetir para cada episodio:
  S <- estado inicial
  A <- epsilon-greedy(S, Q)
  Repetir para cada paso:
    Tomar accion A, observar R, S'
    A' <- epsilon-greedy(S', Q)
    Q(S,A) <- Q(S,A) + alpha * [R + gamma*Q(S',A') - Q(S,A)]
    S <- S', A <- A'
  Hasta que S sea terminal""",
    },
    {
        "name": "Q-Learning",
        "slug": "q-learning",
        "description": """Q-Learning es el algoritmo off-policy de TD control mas conocido. Propuesto por Watkins (1989), actualiza Q(s,a) usando el maximo valor Q del siguiente estado, independientemente de la accion realmente tomada.

Ecuacion de actualizacion:
Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

Al usar max_a para la actualizacion, Q-Learning converge directamente a Q* (la funcion de valor-accion optima) sin importar la politica de comportamiento, siempre que todos los pares (s,a) sean visitados infinitas veces.

Es el fundamento teorico de DQN y muchas variantes modernas de RL.""",
        "category": "td",
        "algorithm_key": "q_learning",
        "icon": "lightning",
        "order": 5,
        "pseudocode": """Inicializar Q(s,a) para todo s,a (ej: 0)

Repetir para cada episodio:
  S <- estado inicial
  Repetir para cada paso:
    A <- epsilon-greedy(S, Q)
    Tomar accion A, observar R, S'
    Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
    S <- S'
  Hasta que S sea terminal""",
    },
    {
        "name": "DQN (Deep Q-Network)",
        "slug": "dqn",
        "description": """Deep Q-Network (Mnih et al., 2015) extiende Q-Learning reemplazando la tabla Q por una red neuronal profunda que aproxima Q(s,a; theta).

Innovaciones clave:
1. Experience Replay: Almacena transiciones (s,a,r,s',done) en un buffer y muestrea mini-batches aleatorios, rompiendo la correlacion temporal.
2. Target Network: Una segunda red theta^- se actualiza periodicamente, estabilizando el entrenamiento al proveer targets fijos.

Loss function:
L(theta) = E[(r + gamma * max_a' Q(s', a'; theta^-) - Q(s, a; theta))^2]

DQN fue el primer algoritmo de RL profundo en alcanzar rendimiento humano en juegos Atari, y es la base de Rainbow, Double DQN, Dueling DQN, y PER.""",
        "category": "deep",
        "algorithm_key": "dqn",
        "icon": "cpu",
        "order": 6,
        "pseudocode": """Inicializar replay buffer D con capacidad N
Inicializar Q-network con pesos theta
Inicializar target network con pesos theta^- = theta

Repetir para cada episodio:
  s <- estado inicial
  Repetir para cada paso:
    Con prob epsilon: a <- accion aleatoria
    Sino: a <- argmax_a Q(s, a; theta)
    Ejecutar a, observar r, s', done
    Almacenar (s, a, r, s', done) en D
    s <- s'

    Muestrear mini-batch de D
    y_j = r_j + gamma * max_a' Q(s'_j, a'; theta^-) * (1 - done_j)
    Minimizar (y_j - Q(s_j, a_j; theta))^2
    Actualizar theta via SGD

  Cada C pasos: theta^- <- theta""",
    },
]


PROBLEMS = [
    # Monte Carlo ES problems
    {
        "name": "Blackjack (21)",
        "slug": "blackjack",
        "env_id": "Blackjack-v1",
        "env_kwargs": {"natural": False, "sab": True},
        "description": "Juego de Blackjack clasico. El agente debe decidir si pedir carta (hit) o plantarse (stick) para acercarse a 21 sin pasarse. Espacio de estados: (suma del jugador, carta visible del dealer, as utilizable). Problema clasico de Sutton & Barto para Monte Carlo.",
        "difficulty": "advanced",
        "state_space_info": "Discreto: (suma: 4-21, carta dealer: 1-10, as usable: bool)",
        "action_space_info": "Discreto: 0=stick, 1=hit",
        "requires_discretization": False,
        "default_episodes": 500000,
        "reward_threshold": 0.0,
        "techniques": ["monte-carlo-es", "monte-carlo-is-ordinario", "monte-carlo-is-ponderado"],
    },
    {
        "name": "FrozenLake 8x8",
        "slug": "frozenlake-8x8",
        "env_id": "FrozenLake-v1",
        "env_kwargs": {"map_name": "8x8", "is_slippery": True},
        "description": "Lago congelado 8x8 con transiciones estocasticas. El agente debe navegar desde la esquina superior izquierda hasta la meta evitando agujeros. Con is_slippery=True, las acciones son no-deterministicas (el agente puede deslizarse). Problema significativamente mas dificil que 4x4.",
        "difficulty": "expert",
        "state_space_info": "Discreto: 64 estados (8x8 grid)",
        "action_space_info": "Discreto: 0=izquierda, 1=abajo, 2=derecha, 3=arriba",
        "requires_discretization": False,
        "default_episodes": 20000,
        "reward_threshold": 0.7,
        "techniques": ["monte-carlo-es", "q-learning"],
    },
    {
        "name": "Cliff Walking",
        "slug": "cliff-walking",
        "env_id": "CliffWalking-v1",
        "env_kwargs": {},
        "description": "Problema clasico del acantilado (Sutton & Barto, Ejemplo 6.6). Grid 4x12 donde el agente debe ir del inicio a la meta bordeando un acantilado. Caer al acantilado da reward -100 y regresa al inicio. Ideal para comparar SARSA (camino seguro) vs Q-Learning (camino optimo pero riesgoso).",
        "difficulty": "advanced",
        "state_space_info": "Discreto: 48 estados (4x12 grid)",
        "action_space_info": "Discreto: 0=arriba, 1=derecha, 2=abajo, 3=izquierda",
        "requires_discretization": False,
        "default_episodes": 5000,
        "reward_threshold": -13,
        "techniques": ["monte-carlo-es", "sarsa", "q-learning"],
    },
    {
        "name": "Taxi-v3",
        "slug": "taxi",
        "env_id": "Taxi-v3",
        "env_kwargs": {},
        "description": "Un taxi en una grilla 5x5 debe recoger un pasajero en una de 4 ubicaciones y dejarlo en otra. 500 estados, 6 acciones. Penalizaciones por movimientos ilegales (-10) y recompensa por entrega exitosa (+20). Problema con espacio de estados moderado pero complejidad combinatoria.",
        "difficulty": "advanced",
        "state_space_info": "Discreto: 500 estados (25 posiciones x 5 estados pasajero x 4 destinos)",
        "action_space_info": "Discreto: 6 acciones (4 movimientos + pickup + dropoff)",
        "requires_discretization": False,
        "default_episodes": 10000,
        "reward_threshold": 8.0,
        "techniques": ["monte-carlo-is-ordinario", "monte-carlo-is-ponderado", "sarsa", "q-learning"],
    },
    {
        "name": "MountainCar (Discretizado)",
        "slug": "mountaincar-disc",
        "env_id": "MountainCar-v0",
        "env_kwargs": {},
        "description": "Un auto debe escalar una montana acumulando momento. El estado continuo (posicion, velocidad) se discretiza. Desafio: sparse reward (solo +0 al llegar, -1 cada paso). Requiere exploracion eficiente y estrategia de momentum. Problema notoriamente dificil para metodos tabulares.",
        "difficulty": "expert",
        "state_space_info": "Continuo discretizado: (posicion [-1.2, 0.6], velocidad [-0.07, 0.07])",
        "action_space_info": "Discreto: 0=izquierda, 1=nada, 2=derecha",
        "requires_discretization": True,
        "discretization_bins": 40,
        "default_episodes": 10000,
        "reward_threshold": -110,
        "techniques": ["sarsa", "q-learning"],
    },
    {
        "name": "Acrobot (Discretizado)",
        "slug": "acrobot-disc",
        "env_id": "Acrobot-v1",
        "env_kwargs": {},
        "description": "Robot de doble pendulo que debe balancear el extremo por encima de una linea. Estado de 6 dimensiones continuas (angulos y velocidades angulares). Discretizado para metodos tabulares. Problema de control clasico con dinamica no lineal compleja.",
        "difficulty": "doctorate",
        "state_space_info": "Continuo discretizado: 6D (cos/sin angulos + velocidades angulares)",
        "action_space_info": "Discreto: 3 acciones (torque -1, 0, +1)",
        "requires_discretization": True,
        "discretization_bins": 10,
        "default_episodes": 10000,
        "reward_threshold": -100,
        "techniques": ["sarsa", "q-learning"],
    },
    {
        "name": "CartPole (Q-Learning Discretizado)",
        "slug": "cartpole-disc",
        "env_id": "CartPole-v1",
        "env_kwargs": {},
        "description": "Equilibrar un pendulo invertido sobre un carro. Estado continuo 4D (posicion, velocidad, angulo, velocidad angular) discretizado para Q-table. Reward +1 por cada paso que el poste se mantiene vertical. Exito: 500 pasos. Clasico benchmark de control.",
        "difficulty": "advanced",
        "state_space_info": "Continuo discretizado: 4D (pos, vel, angulo, vel_angular)",
        "action_space_info": "Discreto: 0=izquierda, 1=derecha",
        "requires_discretization": True,
        "discretization_bins": 20,
        "default_episodes": 10000,
        "reward_threshold": 475,
        "techniques": ["q-learning"],
    },
    {
        "name": "CartPole (DQN)",
        "slug": "cartpole-dqn",
        "env_id": "CartPole-v1",
        "env_kwargs": {},
        "description": "CartPole resuelto con Deep Q-Network. La red neuronal recibe el estado continuo 4D directamente sin discretizacion. Benchmark clasico para validar implementaciones de DQN. Se considera resuelto al mantener reward promedio > 475 durante 100 episodios.",
        "difficulty": "advanced",
        "state_space_info": "Continuo: 4D (posicion, velocidad, angulo, velocidad angular)",
        "action_space_info": "Discreto: 0=izquierda, 1=derecha",
        "requires_discretization": False,
        "default_episodes": 500,
        "reward_threshold": 475,
        "techniques": ["dqn"],
    },
    {
        "name": "LunarLander (DQN)",
        "slug": "lunarlander-dqn",
        "env_id": "LunarLander-v3",
        "env_kwargs": {},
        "description": "Aterrizar una nave espacial en la plataforma de la luna. Estado 8D continuo (posicion, velocidad, angulo, contacto con piernas). 4 acciones discretas. Reward shaping complejo: +100/-100 por aterrizaje/crash, penalizacion por fuel. Problema de nivel avanzado que requiere DQN bien ajustado.",
        "difficulty": "doctorate",
        "state_space_info": "Continuo: 8D (x, y, vx, vy, angulo, vel_angular, pierna_izq, pierna_der)",
        "action_space_info": "Discreto: 0=nada, 1=motor izquierdo, 2=motor principal, 3=motor derecho",
        "requires_discretization": False,
        "default_episodes": 1000,
        "reward_threshold": 200,
        "techniques": ["dqn"],
    },
    {
        "name": "Acrobot (DQN)",
        "slug": "acrobot-dqn",
        "env_id": "Acrobot-v1",
        "env_kwargs": {},
        "description": "Acrobot resuelto con DQN. La red neuronal procesa el estado 6D continuo directamente. Dinamica caoticamente no lineal que requiere aprender una estrategia de swing-up. Problema clasico de control optimo usado en investigacion de nivel doctoral.",
        "difficulty": "doctorate",
        "state_space_info": "Continuo: 6D (cos/sin de 2 angulos + 2 velocidades angulares)",
        "action_space_info": "Discreto: 3 acciones (torque -1, 0, +1)",
        "requires_discretization": False,
        "default_episodes": 500,
        "reward_threshold": -100,
        "techniques": ["dqn"],
    },
    {
        "name": "MountainCar (DQN)",
        "slug": "mountaincar-dqn",
        "env_id": "MountainCar-v0",
        "env_kwargs": {},
        "description": "MountainCar resuelto con DQN. Desafio de sparse reward: el agente solo recibe informacion util al alcanzar la meta. Requiere exploracion eficiente y posiblemente reward shaping o epsilon decay cuidadoso. Problema de referencia en la investigacion de exploracion en RL.",
        "difficulty": "expert",
        "state_space_info": "Continuo: 2D (posicion, velocidad)",
        "action_space_info": "Discreto: 0=izquierda, 1=nada, 2=derecha",
        "requires_discretization": False,
        "default_episodes": 1000,
        "reward_threshold": -110,
        "techniques": ["dqn"],
    },
]


class Command(BaseCommand):
    help = "Carga las tecnicas y problemas de RL en la base de datos"

    def handle(self, *args, **options):
        self.stdout.write("Cargando tecnicas de RL...")

        technique_objs = {}
        for t_data in TECHNIQUES:
            tech, created = Technique.objects.update_or_create(
                slug=t_data["slug"],
                defaults={
                    "name": t_data["name"],
                    "description": t_data["description"],
                    "category": t_data["category"],
                    "algorithm_key": t_data["algorithm_key"],
                    "icon": t_data["icon"],
                    "order": t_data["order"],
                    "pseudocode": t_data["pseudocode"],
                },
            )
            technique_objs[t_data["slug"]] = tech
            status = "creada" if created else "actualizada"
            self.stdout.write(f"  Tecnica {status}: {tech.name}")

        self.stdout.write("\nCargando problemas...")

        for p_data in PROBLEMS:
            technique_slugs = p_data.get("techniques", [])
            problem, created = Problem.objects.update_or_create(
                slug=p_data["slug"],
                env_id=p_data["env_id"],
                defaults={
                    "name": p_data["name"],
                    "env_kwargs": p_data.get("env_kwargs", {}),
                    "description": p_data["description"],
                    "difficulty": p_data["difficulty"],
                    "state_space_info": p_data.get("state_space_info", ""),
                    "action_space_info": p_data.get("action_space_info", ""),
                    "requires_discretization": p_data.get("requires_discretization", False),
                    "discretization_bins": p_data.get("discretization_bins", 20),
                    "default_episodes": p_data.get("default_episodes", 5000),
                    "reward_threshold": p_data.get("reward_threshold"),
                },
            )
            problem.techniques.set([technique_objs[s] for s in technique_slugs if s in technique_objs])
            status = "creado" if created else "actualizado"
            self.stdout.write(f"  Problema {status}: {problem.name} -> {[s for s in technique_slugs]}")

        self.stdout.write(self.style.SUCCESS(
            f"\nSeed completado: {Technique.objects.count()} tecnicas, {Problem.objects.count()} problemas."
        ))
