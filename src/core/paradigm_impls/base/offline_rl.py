"""Offline RL paradigm components.

Wraps the existing LightningBuilder infrastructure for CQL/IQL training
on static transition datasets.
"""

from __future__ import annotations

import logging

from src.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


@register_component("RLReplayBufferDataModule")
class RLReplayBufferDataModule(BaseDataModule):
    """Wraps the existing RLDataModule for offline RL on chunked .pkl datasets."""

    def __init__(self):
        self._data_module = None

    def setup(self, cfg) -> None:
        """Initialize RLDataModule from config."""
        from src.data.rl_data_module import RLDataModule
        self._data_module = RLDataModule(cfg)

    def train_dataloader(self):
        if self._data_module is None:
            raise RuntimeError("Call setup(cfg) before train_dataloader()")
        return self._data_module.train_dataloader()

    def val_dataloader(self):
        if self._data_module is None:
            return None
        return self._data_module.val_dataloader()


@register_component("OfflineRLEvalProtocol")
class OfflineRLEvalProtocol(BaseEvalProtocol):
    """Evaluation protocol for offline RL: val/loss and Bellman error.

    Offline RL evaluation is handled by the Lightning module's validation_step;
    this protocol is a no-op placeholder for the component registry.
    Actual metrics are logged by the Lightning trainer callbacks.
    """

    def evaluate(self, agent, data_source, cfg) -> dict:
        # Offline RL metrics come from Lightning's validation_step / callback_metrics.
        return {}


@register_component("OfflineRLRunner")
class OfflineRLRunner(BaseParadigmRunner):
    """Runs the offline RL training loop for each (dataset, agent) pair.

    Calls run_offline_phase() directly from local_runner, which subprocesses
    train.py for each agent×dataset combination with the correct Hydra overrides.
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
            run_plotting_phase,
        )

        _setup_output_dirs(cfg)

        if not cfg.get("no_offline", False):
            # No prior online phase — pass empty trial ids so dataset resolution
            # falls back to the configured yaml_ds_path / offline_datasets list.
            run_offline_phase(cfg, context, best_online_trial_ids={})
        else:
            log.info("Skipping offline training phase (no_offline=true).")

        if not cfg.get("no_plot", False):
            run_plotting_phase(cfg, context)
