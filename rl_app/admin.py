from django.contrib import admin
from .models import Technique, Problem, TrainingRun


@admin.register(Technique)
class TechniqueAdmin(admin.ModelAdmin):
    list_display = ["name", "category", "algorithm_key", "order"]
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Problem)
class ProblemAdmin(admin.ModelAdmin):
    list_display = ["name", "env_id", "difficulty", "requires_discretization"]
    list_filter = ["difficulty", "techniques"]
    prepopulated_fields = {"slug": ("name",)}
    filter_horizontal = ["techniques"]


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ["technique", "problem", "status", "best_reward", "execution_time", "created_at"]
    list_filter = ["status", "technique", "problem"]
    readonly_fields = ["created_at", "completed_at"]
