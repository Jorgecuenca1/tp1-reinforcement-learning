from django.urls import path
from . import views

app_name = "rl_app"

urlpatterns = [
    path("", views.home, name="home"),
    path("tecnica/<slug:slug>/", views.technique_detail, name="technique_detail"),
    path("entrenar/<slug:technique_slug>/<slug:problem_slug>/", views.run_training, name="run_training"),
    path("resultado/<int:pk>/", views.results, name="results"),
    path("historial/", views.results_list, name="results_list"),
    path("exportar/<int:pk>/", views.export_colab, name="export_colab"),
    path("comparar/", views.compare, name="compare"),
    path("api/run/", views.api_run_training, name="api_run_training"),
    path("snake/", views.snake_visual, name="snake_visual"),
    path("predator-prey/", views.predator_prey_visual, name="predator_prey"),
]
