"""Standard RL task — dispatches to the paradigm-defined runner.

Registered as the default 'rl' task prefix. Dispatched by run_pipeline.py
when task_name is 'rl' (or unset).

Local execution path:
  1. Read paradigm_def from context (loaded by run_pipeline.py).
  2. Instantiate data_module, eval_protocol, and runner from paradigm_def.
  3. Call runner.run(cfg, data_module, eval_protocol, callbacks, context).
  4. Fallback to run_local_training() if paradigm_def failed to load.

Slurm execution always goes through run_slurm_training() regardless of paradigm,
since cluster submission logic is environment-independent.
"""

import logging

from src.app.pipeline.task_registry import register_task

log = logging.getLogger(__name__)


@register_task("rl")
def run_standard_rl_task(cfg, context):
    """Execute the RL pipeline via the paradigm-defined runner."""
    paradigm_def = context.get("paradigm_def")

    if context["is_interactive"]:
        if paradigm_def and paradigm_def.runner_cls is not None:
            log.info(
                "Paradigm '%s': dispatching to %s.",
                paradigm_def.name,
                paradigm_def.runner_cls.__name__,
            )
            data_module = (
                paradigm_def.data_module_cls() if paradigm_def.data_module_cls else None
            )
            eval_protocol = (
                paradigm_def.eval_protocol_cls() if paradigm_def.eval_protocol_cls else None
            )
            runner = paradigm_def.runner_cls()
            runner.run(cfg, data_module, eval_protocol, [], context)
        else:
            raise RuntimeError(f"No paradigm runner resolved (paradigm_def={paradigm_def}).")
    else:
        from src.app.pipeline.slurm_runner import run_slurm_training

        run_slurm_training(cfg, context)
