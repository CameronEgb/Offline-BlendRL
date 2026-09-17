"""Standard RL task — online + offline training phases.

Registered as the default 'rl' task prefix. Dispatched by run_pipeline.py
when task_name is 'rl' (or unset).
"""

from src.pipeline.task_registry import register_task


@register_task("rl")
def run_standard_rl_task(cfg, context):
    """Execute the standard RL pipeline: online training → offline training → plotting."""
    if context["is_interactive"]:
        from src.pipeline.local_runner import run_local_training

        run_local_training(cfg, context)
    else:
        from src.pipeline.slurm_runner import run_slurm_training

        run_slurm_training(cfg, context)
