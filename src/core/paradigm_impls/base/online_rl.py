"""Online RL paradigm components.

Wraps the existing online training loop (VectorizedEnv + Lightning) for
simulator-based RL agents (PPO, BlendRL online).
"""

from __future__ import annotations

import logging

from src.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


@register_component("SimulatorDataModule")
class SimulatorDataModule(BaseDataModule):
    """Data 'source' for online RL: the environment itself generates transitions.

    Online RL doesn't use a static dataset; this is a stub to satisfy the interface.
    The actual rollout loop is owned by OnlineRLRunner via train.py.
    """

    def setup(self, cfg) -> None:
        pass  # Environment setup handled inside train.py

    def train_dataloader(self):
        raise NotImplementedError(
            "Online RL generates data via environment rollouts, not a static DataLoader. "
            "Use OnlineRLRunner.run() which calls the online training phase directly."
        )


@register_component("EpisodicRewardEvalProtocol")
class EpisodicRewardEvalProtocol(BaseEvalProtocol):
    """Evaluation protocol for online RL: fixed-episode rollouts, mean reward.

    Actual evaluation is handled by EnvironmentEvaluatorCallback inside
    the Lightning training loop. This is a registry placeholder.
    """

    def evaluate(self, agent, data_source, cfg) -> dict:
        # Online eval is handled by EnvironmentEvaluatorCallback during trainer.fit().
        return {}


@register_component("OnlineRLRunner")
class OnlineRLRunner(BaseParadigmRunner):
    """Runs the online RL training loop, then optionally trains offline agents.

    Calls run_online_phase() for each declared online agent. If offline_methods
    are also configured in the experiment, calls run_offline_phase() afterwards
    so that a single online_rl experiment can collect data and bench offline agents
    against it in one pipeline run.
    """

    def run(
        self,
        cfg,
        data_module: BaseDataModule,
        eval_protocol: BaseEvalProtocol,
        callbacks: list,
        context: dict,
    ) -> None:
        from src.pipeline.local_runner import (
            _setup_output_dirs,
            run_offline_phase,
            run_online_phase,
            run_plotting_phase,
        )

        _setup_output_dirs(cfg)

        best_online_trial_ids: dict = {}
        if not cfg.get("no_online", False):
            best_online_trial_ids = run_online_phase(cfg, context)
        else:
            log.info("Skipping online training phase (no_online=true).")

        # If offline agents are configured, train them on the collected datasets.
        offline_list = context.get("offline_list", [])
        if offline_list and not cfg.get("no_offline", False):
            log.info("Offline methods detected; running offline phase after online collection.")
            run_offline_phase(cfg, context, best_online_trial_ids)
        elif offline_list:
            log.info("Skipping offline training phase (no_offline=true).")

        if not cfg.get("no_plot", False):
            run_plotting_phase(cfg, context)
