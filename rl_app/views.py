import json
import time
import os
import traceback
import numpy as np
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.conf import settings
from .models import Technique, Problem, TrainingRun
from .algorithms import monte_carlo_es, monte_carlo_is, sarsa, q_learning, dqn, snake_dqn, predator_prey_dqn
from .algorithms.utils import plot_convergence


ALGORITHM_MAP = {
    "monte_carlo_es": monte_carlo_es,
    "monte_carlo_is_ordinary": monte_carlo_is,
    "monte_carlo_is_weighted": monte_carlo_is,
    "sarsa": sarsa,
    "q_learning": q_learning,
    "dqn": dqn,
}


def home(request):
    techniques = Technique.objects.prefetch_related("problems").all()
    recent_runs = TrainingRun.objects.select_related("technique", "problem")[:8]
    stats = {
        "total_runs": TrainingRun.objects.filter(status="completed").count(),
        "techniques_count": Technique.objects.count(),
        "problems_count": Problem.objects.count(),
    }
    return render(request, "rl_app/home.html", {
        "techniques": techniques,
        "recent_runs": recent_runs,
        "stats": stats,
    })


def technique_detail(request, slug):
    technique = get_object_or_404(Technique, slug=slug)
    problems = technique.problems.all()
    runs = TrainingRun.objects.filter(technique=technique, status="completed").order_by("-created_at")[:5]
    return render(request, "rl_app/technique_detail.html", {
        "technique": technique,
        "problems": problems,
        "recent_runs": runs,
    })


def run_training(request, technique_slug, problem_slug):
    technique = get_object_or_404(Technique, slug=technique_slug)
    problem = get_object_or_404(Problem, slug=problem_slug, techniques=technique)

    if request.method == "POST":
        run = TrainingRun(technique=technique, problem=problem)
        run.num_episodes = int(request.POST.get("num_episodes", problem.default_episodes))
        run.gamma = float(request.POST.get("gamma", 0.99))
        run.alpha = float(request.POST.get("alpha", 0.1))
        run.epsilon = float(request.POST.get("epsilon", 1.0))
        run.epsilon_decay = float(request.POST.get("epsilon_decay", 0.9995))
        run.epsilon_min = float(request.POST.get("epsilon_min", 0.01))
        run.hidden_layers = request.POST.get("hidden_layers", "128,128")
        run.batch_size = int(request.POST.get("batch_size", 64))
        run.target_update = int(request.POST.get("target_update", 10))
        run.replay_buffer_size = int(request.POST.get("replay_buffer_size", 10000))
        run.learning_rate_dqn = float(request.POST.get("learning_rate_dqn", 0.001))
        run.status = "running"
        run.save()

        try:
            algo_key = technique.algorithm_key
            algo_module = ALGORITHM_MAP.get(algo_key)
            if not algo_module:
                raise ValueError(f"Algoritmo no encontrado: {algo_key}")

            train_kwargs = {
                "env_id": problem.env_id,
                "env_kwargs": problem.env_kwargs or {},
                "num_episodes": run.num_episodes,
                "gamma": run.gamma,
                "alpha": run.alpha,
                "epsilon": run.epsilon,
                "epsilon_decay": run.epsilon_decay,
                "epsilon_min": run.epsilon_min,
                "requires_discretization": problem.requires_discretization,
                "discretization_bins": problem.discretization_bins,
            }

            if algo_key in ("monte_carlo_is_ordinary", "monte_carlo_is_weighted"):
                train_kwargs["variant"] = "ordinary" if "ordinary" in algo_key else "weighted"

            if algo_key == "dqn":
                train_kwargs["hidden_layers"] = run.hidden_layers
                train_kwargs["batch_size"] = run.batch_size
                train_kwargs["target_update"] = run.target_update
                train_kwargs["replay_buffer_size"] = run.replay_buffer_size
                train_kwargs["learning_rate_dqn"] = run.learning_rate_dqn

            start_time = time.time()
            rewards, metrics = algo_module.train(**train_kwargs)
            elapsed = time.time() - start_time

            run.set_rewards([float(r) for r in rewards])
            run.execution_time = round(elapsed, 2)
            run.best_reward = float(max(rewards)) if rewards else 0
            run.avg_reward_last_100 = float(np.mean(rewards[-100:])) if rewards else 0
            run.total_steps = sum(1 for _ in rewards)
            run.q_table_size = metrics.get("q_table_size")
            run.extra_metrics = metrics

            graph_filename = f"run_{run.pk}.png"
            graph_path = os.path.join(settings.MEDIA_ROOT, "graphs", graph_filename)
            title = f"{technique.name} - {problem.name}"
            plot_convergence(rewards, run.moving_avg_window, title, graph_path)
            run.convergence_graph = f"graphs/{graph_filename}"

            run.status = "completed"
            run.completed_at = timezone.now()
            run.save()

            return redirect("rl_app:results", pk=run.pk)

        except Exception as e:
            run.status = "failed"
            run.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
            run.completed_at = timezone.now()
            run.save()
            return redirect("rl_app:results", pk=run.pk)

    defaults = {
        "num_episodes": problem.default_episodes,
        "gamma": 0.99,
        "alpha": 0.5 if technique.algorithm_key in ("sarsa", "q_learning") else 0.1,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "hidden_layers": "128,128",
        "batch_size": 64,
        "target_update": 10,
        "replay_buffer_size": 10000,
        "learning_rate_dqn": 0.001,
    }

    return render(request, "rl_app/run_training.html", {
        "technique": technique,
        "problem": problem,
        "defaults": defaults,
        "is_dqn": technique.algorithm_key == "dqn",
        "is_td": technique.algorithm_key in ("sarsa", "q_learning"),
    })


def results(request, pk):
    run = get_object_or_404(TrainingRun.objects.select_related("technique", "problem"), pk=pk)
    rewards = run.get_rewards()

    chart_data = None
    if rewards:
        window = run.moving_avg_window
        moving_avg = []
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            moving_avg = [round(float(x), 2) for x in ma]

        chart_data = {
            "labels": list(range(1, len(rewards) + 1)),
            "rewards": [round(float(r), 2) for r in rewards],
            "moving_avg": moving_avg,
            "ma_start": window,
        }

    return render(request, "rl_app/results.html", {
        "run": run,
        "chart_data": json.dumps(chart_data) if chart_data else None,
    })


def results_list(request):
    runs = TrainingRun.objects.select_related("technique", "problem").all()
    technique_filter = request.GET.get("technique")
    status_filter = request.GET.get("status")
    if technique_filter:
        runs = runs.filter(technique__slug=technique_filter)
    if status_filter:
        runs = runs.filter(status=status_filter)

    techniques = Technique.objects.all()
    return render(request, "rl_app/results_list.html", {
        "runs": runs,
        "techniques": techniques,
        "current_technique": technique_filter,
        "current_status": status_filter,
    })


def export_colab(request, pk):
    run = get_object_or_404(TrainingRun.objects.select_related("technique", "problem"), pk=pk)

    algo_key = run.technique.algorithm_key
    algo_module = ALGORITHM_MAP.get(algo_key)

    hyperparams = {
        "num_episodes": run.num_episodes,
        "gamma": run.gamma,
        "alpha": run.alpha,
        "epsilon": run.epsilon,
        "epsilon_decay": run.epsilon_decay,
        "epsilon_min": run.epsilon_min,
        "requires_discretization": run.problem.requires_discretization,
        "discretization_bins": run.problem.discretization_bins,
    }

    if algo_key == "dqn":
        hl = [int(x.strip()) for x in run.hidden_layers.split(",") if x.strip()]
        hyperparams["hidden_layers"] = hl
        hyperparams["batch_size"] = run.batch_size
        hyperparams["target_update"] = run.target_update
        hyperparams["replay_buffer_size"] = run.replay_buffer_size
        hyperparams["learning_rate_dqn"] = run.learning_rate_dqn

    if algo_key in ("monte_carlo_is_ordinary", "monte_carlo_is_weighted"):
        hyperparams["variant"] = "ordinary" if "ordinary" in algo_key else "weighted"

    code = algo_module.get_colab_code(
        run.problem.env_id,
        run.problem.env_kwargs,
        hyperparams,
        run.problem.name,
    )

    filename = f"{algo_key}_{run.problem.slug}_{run.pk}.py"
    response = HttpResponse(code, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def compare(request):
    run_ids = request.GET.getlist("runs")
    runs = []
    chart_datasets = []
    colors = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ffd700", "#ff7b72"]

    if run_ids:
        for i, rid in enumerate(run_ids[:6]):
            try:
                run = TrainingRun.objects.select_related("technique", "problem").get(pk=int(rid))
                runs.append(run)
                rewards = run.get_rewards()
                if rewards:
                    window = run.moving_avg_window
                    if len(rewards) >= window:
                        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
                        chart_datasets.append({
                            "label": f"{run.technique.name} - {run.problem.name}",
                            "data": [round(float(x), 2) for x in ma],
                            "color": colors[i % len(colors)],
                            "start": window,
                        })
            except (TrainingRun.DoesNotExist, ValueError):
                continue

    all_runs = TrainingRun.objects.filter(status="completed").select_related("technique", "problem").order_by("-created_at")[:50]

    return render(request, "rl_app/compare.html", {
        "runs": runs,
        "all_runs": all_runs,
        "chart_datasets": json.dumps(chart_datasets),
        "selected_ids": [int(x) for x in run_ids if x.isdigit()],
    })


def api_run_training(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    return JsonResponse({"status": "use form submit instead"})


def snake_visual(request):
    """Vista para entrenar y visualizar Snake con DQN."""
    replay_data = None
    chart_data = None
    run_info = None
    error = None

    if request.method == "POST":
        try:
            num_episodes = int(request.POST.get("num_episodes", 500))
            grid_size = int(request.POST.get("grid_size", 10))
            epsilon_decay = float(request.POST.get("epsilon_decay", 0.997))
            hidden_size = int(request.POST.get("hidden_size", 256))
            learning_rate = float(request.POST.get("learning_rate", 0.001))

            start_time = time.time()
            rewards, metrics = snake_dqn.train(
                grid_size=grid_size,
                num_episodes=num_episodes,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=epsilon_decay,
                epsilon_min=0.01,
                batch_size=64,
                target_update=10,
                replay_buffer_size=50000,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
            )
            elapsed = time.time() - start_time

            replay_data = json.dumps({
                "frames": metrics["replay_frames"],
                "grid_size": metrics["grid_size"],
                "score": metrics["replay_score"],
            })

            # Replays guardados del entrenamiento (sin frames para el listado)
            saved_replays_meta = []
            saved_replays_full = metrics.get("saved_replays", [])
            for sr in saved_replays_full:
                saved_replays_meta.append({
                    "episode": sr["episode"],
                    "score": sr["score"],
                    "reward": sr["reward"],
                    "frames_count": len(sr["frames"]),
                })
            all_replays_data = json.dumps([
                {"episode": sr["episode"], "score": sr["score"], "reward": sr["reward"],
                 "frames": sr["frames"]}
                for sr in saved_replays_full
            ])

            window = min(100, len(rewards) // 3) if len(rewards) > 30 else 10
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            scores = metrics.get("scores_history", [])
            scores_ma = []
            if len(scores) >= window:
                scores_ma = list(np.convolve(scores, np.ones(window) / window, mode='valid'))

            chart_data = json.dumps({
                "labels": list(range(1, len(rewards) + 1)),
                "rewards": [round(float(r), 2) for r in rewards],
                "moving_avg": [round(float(x), 2) for x in moving_avg],
                "ma_start": window,
                "scores": scores,
                "scores_ma": [round(float(x), 2) for x in scores_ma],
            })

            graph_path = os.path.join(settings.MEDIA_ROOT, "graphs", "snake_latest.png")
            plot_convergence(rewards, window, f"DQN Snake {grid_size}x{grid_size}", graph_path)

            run_info = {
                "best_score": metrics["best_score"],
                "avg_score_last_100": metrics["avg_score_last_100"],
                "max_score_eval": metrics["max_score_eval"],
                "replay_score": metrics["replay_score"],
                "episodes": num_episodes,
                "time": round(elapsed, 1),
                "params": metrics["network_params"],
                "grid_size": grid_size,
            }

        except Exception as e:
            error = f"{str(e)}\n{traceback.format_exc()}"

    return render(request, "rl_app/snake_visual.html", {
        "replay_data": replay_data,
        "chart_data": chart_data,
        "run_info": run_info,
        "saved_replays_meta": json.dumps(saved_replays_meta) if run_info else "[]",
        "all_replays_data": all_replays_data if run_info else "[]",
        "error": error,
    })


def predator_prey_visual(request):
    """Multi-Agent RL: Predator-Prey con visualizacion."""
    replay_data = None
    chart_data = None
    run_info = None
    error = None

    if request.method == "POST":
        try:
            num_episodes = int(request.POST.get("num_episodes", 500))
            grid_size = int(request.POST.get("grid_size", 12))
            n_predators = int(request.POST.get("n_predators", 3))
            n_prey = int(request.POST.get("n_prey", 1))
            vision_range = int(request.POST.get("vision_range", 4))
            max_steps_per_ep = int(request.POST.get("max_steps", 200))
            epsilon_decay = float(request.POST.get("epsilon_decay", 0.998))
            hidden_size = int(request.POST.get("hidden_size", 256))
            learning_rate = float(request.POST.get("learning_rate", 0.0005))

            start_time = time.time()
            rewards, metrics = predator_prey_dqn.train(
                grid_size=grid_size,
                n_predators=n_predators,
                n_prey=n_prey,
                vision_range=vision_range,
                num_episodes=num_episodes,
                max_steps_per_ep=max_steps_per_ep,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=epsilon_decay,
                epsilon_min=0.05,
                batch_size=128,
                target_update=15,
                replay_buffer_size=100000,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
            )
            elapsed = time.time() - start_time

            replay_data = json.dumps({
                "frames": metrics["replay_frames"],
                "grid_size": metrics["grid_size"],
                "n_predators": metrics["n_predators"],
                "n_prey": metrics["n_prey"],
            })

            window = min(100, len(rewards) // 3) if len(rewards) > 30 else 10
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

            captures = metrics.get("captures_history", [])
            captures_cumsum = list(np.cumsum(captures))
            cap_rate = []
            for idx in range(len(captures)):
                start_idx = max(0, idx - 99)
                cap_rate.append(round(sum(1 for c in captures[start_idx:idx + 1] if c > 0)
                                      / (idx - start_idx + 1) * 100, 1))

            chart_data = json.dumps({
                "labels": list(range(1, len(rewards) + 1)),
                "rewards": [round(float(r), 2) for r in rewards],
                "moving_avg": [round(float(x), 2) for x in moving_avg],
                "ma_start": window,
                "capture_rate": cap_rate,
                "captures_cumulative": captures_cumsum,
            })

            graph_path = os.path.join(settings.MEDIA_ROOT, "graphs", "predator_prey_latest.png")
            plot_convergence(rewards, window,
                             f"MARL Predator-Prey ({n_predators}P vs {n_prey}E) {grid_size}x{grid_size}",
                             graph_path)

            saved_replays_full = metrics.get("saved_replays", [])
            saved_replays_meta = []
            for sr in saved_replays_full:
                saved_replays_meta.append({
                    "episode": sr["episode"],
                    "captures": sr["captures"],
                    "steps": sr["steps"],
                    "reward": sr["reward"],
                    "frames_count": len(sr["frames"]),
                })
            all_replays_data = json.dumps([
                {"episode": sr["episode"], "captures": sr["captures"],
                 "steps": sr["steps"], "reward": sr["reward"],
                 "frames": sr["frames"]}
                for sr in saved_replays_full
            ])

            run_info = {
                "capture_rate_total": metrics["capture_rate_total"],
                "capture_rate_last_100": metrics["capture_rate_last_100"],
                "eval_capture_rate": metrics["eval_capture_rate"],
                "avg_steps_to_capture": metrics["avg_steps_to_capture"],
                "best_capture_time": metrics["best_capture_time"],
                "total_cooperation": metrics["total_cooperation_events"],
                "episodes": num_episodes,
                "time": round(elapsed, 1),
                "params": metrics["network_params"],
                "grid_size": grid_size,
                "n_predators": n_predators,
                "n_prey": n_prey,
            }

        except Exception as e:
            error = f"{str(e)}\n{traceback.format_exc()}"

    return render(request, "rl_app/predator_prey.html", {
        "replay_data": replay_data,
        "chart_data": chart_data,
        "run_info": run_info,
        "saved_replays_meta": json.dumps(saved_replays_meta) if run_info else "[]",
        "all_replays_data": all_replays_data if run_info else "[]",
        "error": error,
    })
