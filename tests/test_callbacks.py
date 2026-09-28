"""EnvironmentEvaluatorCallback logging."""

from types import SimpleNamespace

from src.app.core.callbacks import EnvironmentEvaluatorCallback


class RecordingLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((dict(metrics), step))


def test_evaluation_metrics_reach_every_logger(monkeypatch):
    csv_logger, tensorboard_logger = RecordingLogger(), RecordingLogger()
    trainer = SimpleNamespace(global_step=40, loggers=[csv_logger, tensorboard_logger], logger=csv_logger,
                              fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=5)))
    logged = []
    pl_module = SimpleNamespace(cfg=SimpleNamespace(paradigm="online_rl"),
                                log=lambda name, value, **kwargs: logged.append(name))

    callback = EnvironmentEvaluatorCallback(cfg=None)
    callback.train_start_time = 0.0
    monkeypatch.setattr(callback, "evaluate", lambda trainer, pl_module: (25.0, 4.0))
    callback.evaluate_and_log(trainer, pl_module, transitions=10240)

    for recorded in (csv_logger, tensorboard_logger):
        assert len(recorded.calls) == 1
        metrics, step = recorded.calls[0]
        assert step == 4  # Lightning's own epoch-end step, not global_step (optimizer steps)
        assert (metrics["eval/reward"], metrics["eval/reward_std"], metrics["transitions"]) == (25.0, 4.0, 10240.0)
        assert {"time/eval", "time/train", "time/total"} <= metrics.keys()
    # Still reported to Lightning for checkpoint monitoring and the progress bar
    assert "eval/reward" in logged


def test_log_step_matches_lightning_and_falls_back_to_global_step():
    step = EnvironmentEvaluatorCallback._lightning_log_step
    loop = lambda n: SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=n))  # noqa: E731
    assert step(SimpleNamespace(global_step=0, fit_loop=loop(0)), 0) == 0  # evaluation before training
    assert step(SimpleNamespace(global_step=32, fit_loop=loop(4)), 2048) == 3
    assert step(SimpleNamespace(global_step=32), 2048) == 32  # attribute renamed in a future Lightning
