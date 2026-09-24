"""Online RL paradigm components.

Wraps the existing online training loop (VectorizedEnv + Lightning) for
simulator-based RL agents (PPO, BlendRL online).
"""

from __future__ import annotations

import logging

from src.app.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.app.core.paradigm_loader import register_component

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
    """Runs the online RL training loop for simulator-based agents.

    Dispatches declared online methods (e.g. PPO, BlendRL online) sequentially
    or via cluster jobs, followed by the automated plotting phase.
    """

    def run(
        self,
        cfg,
        data_module: BaseDataModule,
        eval_protocol: BaseEvalProtocol,
        callbacks: list,
        context: dict,
    ) -> None:
        from src.app.pipeline.local_runner import (
            _setup_output_dirs,
            run_methods,
            run_plotting_phase,
        )

        _setup_output_dirs(cfg)
        run_methods(cfg, context)

        if not cfg.get("no_plot", False):
            run_plotting_phase(cfg, context)
