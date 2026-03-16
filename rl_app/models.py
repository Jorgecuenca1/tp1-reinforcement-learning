from django.db import models
import json


class Technique(models.Model):
    CATEGORY_CHOICES = [
        ("monte_carlo", "Monte Carlo"),
        ("td", "Temporal Difference"),
        ("deep", "Deep Learning"),
    ]
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    algorithm_key = models.CharField(max_length=50)
    icon = models.CharField(max_length=20, default="cpu")
    order = models.IntegerField(default=0)
    pseudocode = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "name"]

    def __str__(self):
        return self.name


class Problem(models.Model):
    DIFFICULTY_CHOICES = [
        ("intermediate", "Intermedio"),
        ("advanced", "Avanzado"),
        ("expert", "Experto"),
        ("doctorate", "Doctorado"),
    ]
    name = models.CharField(max_length=200)
    slug = models.SlugField()
    env_id = models.CharField(max_length=100)
    env_kwargs = models.JSONField(default=dict, blank=True)
    description = models.TextField()
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES)
    techniques = models.ManyToManyField(Technique, related_name="problems")
    state_space_info = models.TextField(blank=True)
    action_space_info = models.TextField(blank=True)
    requires_discretization = models.BooleanField(default=False)
    discretization_bins = models.IntegerField(default=20)
    default_episodes = models.IntegerField(default=5000)
    reward_threshold = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        unique_together = ["slug", "env_id"]

    def __str__(self):
        return f"{self.name} ({self.env_id})"

    def get_env_kwargs_display(self):
        if self.env_kwargs:
            return ", ".join(f"{k}={v}" for k, v in self.env_kwargs.items())
        return "Default"


class TrainingRun(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pendiente"),
        ("running", "Ejecutando"),
        ("completed", "Completado"),
        ("failed", "Fallido"),
    ]
    technique = models.ForeignKey(Technique, on_delete=models.CASCADE, related_name="runs")
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE, related_name="runs")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    # Hyperparameters
    num_episodes = models.IntegerField(default=5000)
    gamma = models.FloatField(default=0.99)
    alpha = models.FloatField(default=0.1)
    epsilon = models.FloatField(default=1.0)
    epsilon_decay = models.FloatField(default=0.9995)
    epsilon_min = models.FloatField(default=0.01)
    # DQN specific
    hidden_layers = models.CharField(max_length=100, default="128,128")
    batch_size = models.IntegerField(default=64)
    target_update = models.IntegerField(default=10)
    replay_buffer_size = models.IntegerField(default=10000)
    learning_rate_dqn = models.FloatField(default=0.001)
    # Results
    rewards_json = models.TextField(default="[]")
    moving_avg_window = models.IntegerField(default=100)
    best_reward = models.FloatField(null=True, blank=True)
    avg_reward_last_100 = models.FloatField(null=True, blank=True)
    total_steps = models.IntegerField(null=True, blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    convergence_graph = models.ImageField(upload_to="graphs/", null=True, blank=True)
    q_table_size = models.IntegerField(null=True, blank=True)
    extra_metrics = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.technique.name} - {self.problem.name} ({self.status})"

    def get_rewards(self):
        try:
            return json.loads(self.rewards_json)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_rewards(self, rewards_list):
        self.rewards_json = json.dumps(rewards_list)

    @property
    def duration_display(self):
        if self.execution_time:
            if self.execution_time < 60:
                return f"{self.execution_time:.1f}s"
            return f"{self.execution_time / 60:.1f}min"
        return "-"
