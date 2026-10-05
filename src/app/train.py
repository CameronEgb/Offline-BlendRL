import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for p in [
    PROJECT_ROOT,
    SRC_DIR,
    os.path.join(SRC_DIR, "app"),
    os.path.join(SRC_DIR, "usr"),
    os.path.join(SRC_DIR, "usr", "models"),
    os.path.join(SRC_DIR, "usr", "environments"),
    os.path.join(SRC_DIR, "usr", "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
import time
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    safe_types = [
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.Container,
    ]
    for node_name in ["AnyNode", "Node", "ValueNode", "UntypedNode"]:
        if hasattr(omegaconf.nodes, node_name):
            safe_types.append(getattr(omegaconf.nodes, node_name))
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_types)
except (AttributeError, TypeError) as e:
    logger.debug("PyTorch safe globals registration skipped: %s", e)
except Exception as e:
    logger.debug("Unexpected error registering safe globals: %s", e)

from src.app.core.lightning_builder import build_trainer, finalize_training
from src.app.data.rl_data_module import RLDataModule
from src.usr.methods.agent_registry import auto_discover, get_agent_class


@hydra.main(version_base=None, config_path="../../in/config", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("experiment_id", "default_exp") == "default_exp":
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                for override in HydraConfig.get().overrides.task:
                    if override.startswith("+experiment=") or override.startswith("experiment="):
                        exp_stem = Path(override.split("=")[-1]).stem
                        cfg.experiment_id = exp_stem
                        break
        except Exception as e:
            logger.debug("Could not infer experiment_id from Hydra task overrides: %s", e)

    print(OmegaConf.to_yaml(cfg))

    # === OS ENVIRON BRIDGE PATTERN ===
    # We copy certain Hydra config values into os.environ to pass them down to nested components
    # (like gym environments or legacy hooks) that cannot easily receive the `cfg` object directly.
    # While some components have been refactored to read from `cfg`, others still rely on this.
    if "env" in cfg and "reward_type" in cfg.env:
        os.environ["MIMIC_REWARD_TYPE"] = str(cfg.env.reward_type)

    paradigm_name = cfg.get("paradigm", "online_rl")
    from src.app.core.paradigm_loader import get_component, load_paradigm_definition

    paradigm_def = load_paradigm_definition(paradigm_name)

    # 1. Resolve Data Module from explicit config or paradigm definition
    dm_name = cfg.get("data_module") or (cfg.env.get("data_module") if hasattr(cfg, "env") else None)
    DataModuleCls = get_component(dm_name) if dm_name else paradigm_def.data_module_cls
    if DataModuleCls is None:
        DataModuleCls = RLDataModule

    datamodule = DataModuleCls(cfg)

    # 2. Build Model / Agent based on paradigm
    if paradigm_name == "supervised":
        input_dim = getattr(datamodule, "input_dim", 64)

        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else getattr(cfg, "model", {})
        arch_name = str(model_cfg.get("architecture", model_cfg.get("name", "lstm"))).lower()
        lr = float(model_cfg.get("lr", cfg.get("lr", 1e-3)))

        # Check if an explicit LightningModule component was registered for this architecture
        target_module_name = model_cfg.get("lightning_module") or model_cfg.get("module")
        SupervisedModelCls = None
        if target_module_name:
            try:
                SupervisedModelCls = get_component(target_module_name)
            except KeyError:
                SupervisedModelCls = None
        if SupervisedModelCls is None:
            from src.usr.eval.early_prediction.lightning_module import EPSepsisLightningModule

            SupervisedModelCls = EPSepsisLightningModule

        kwargs = {}
        if hasattr(model_cfg, "items"):
            reserved = {"architecture", "name", "lightning_module", "module", "lr", "type", "epochs_per_interval", "eval_interval_epochs"}
            for k, v in model_cfg.items():
                if k not in reserved and v is not None:
                    kwargs[k] = v

        for k in (
            "hidden_dim",
            "num_layers",
            "dropout",
            "use_dual_pooling",
            "use_tcn_conv",
            "bidirectional",
            "d_model",
            "nhead",
            "dim_feedforward",
            "pos_type",
            "max_len",
            "use_cls_token",
            "use_focal_loss",
            "pos_weight",
            "weight_decay",
        ):
            if k not in kwargs and hasattr(cfg, "get") and cfg.get(k) is not None:
                kwargs[k] = cfg.get(k)

        print(f"Supervised Paradigm: constructing {SupervisedModelCls.__name__} ({arch_name.upper()}, input_dim={input_dim}, lr={lr})")
        model = SupervisedModelCls(architecture_name=arch_name, input_dim=input_dim, lr=lr, **kwargs)
    else:
        auto_discover()
        agent_cfg = cfg.agent
        base_algo_name = agent_cfg.get("algorithm", agent_cfg.get("name", None))
        print(f"Extracted algorithm name: {base_algo_name}")

        if not base_algo_name:
            raise ValueError("Could not extract algorithm name from config.")

        AgentClass = get_agent_class(base_algo_name)
        print(f"Resolved agent class: {AgentClass.__name__}")
        model = AgentClass(cfg)

    trainer, ckpt_dir, ckpt_path = build_trainer(cfg, model)

    start_time = time.time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n[Training Complete] Total execution time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    metric = finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time)

    if paradigm_name == "supervised":
        _run_supervised_evaluation(cfg, model, datamodule, paradigm_def)

    return metric


def _run_supervised_evaluation(cfg, model, datamodule, paradigm_def):
    """Post-training horizon evaluation for supervised tasks (e.g. Sepsis lead-time sweep)."""
    import json
    from pathlib import Path
    from src.app.core.lightning_builder import get_method_name

    delegate = getattr(datamodule, "_delegate", datamodule)
    if not hasattr(delegate, "get_eval_dataloader"):
        return

    ep_cfg = cfg.get("early_prediction", {}) if hasattr(cfg, "get") else {}
    if hasattr(ep_cfg, "__iter__") and not isinstance(ep_cfg, dict):
        from omegaconf import OmegaConf

        ep_cfg = OmegaConf.to_container(ep_cfg, resolve=True)
    if not isinstance(ep_cfg, dict):
        ep_cfg = {}

    eval_cfg = cfg.get("eval_protocol", {}) if hasattr(cfg, "get") else {}
    if hasattr(eval_cfg, "__iter__") and not isinstance(eval_cfg, dict):
        from omegaconf import OmegaConf

        eval_cfg = OmegaConf.to_container(eval_cfg, resolve=True)
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}

    horizons = eval_cfg.get("horizons")
    if not horizons:
        tau_min = ep_cfg.get("tau_min")
        tau_max = ep_cfg.get("tau_max")
        tau_step = ep_cfg.get("tau_step", 1)
        if tau_min is not None and tau_max is not None:
            horizons = list(range(int(tau_min), int(tau_max) + 1, int(tau_step)))
    if not horizons:
        horizons = [1, 5, 9, 13, 17, 21, 25, 29, 33]

    eval_protocol_cls = (
        paradigm_def.eval_protocol_cls
        if (paradigm_def and paradigm_def.eval_protocol_cls)
        else None
    )
    if eval_protocol_cls is None:
        from src.app.core.paradigm_impls.base.supervised import ClassificationEvalProtocol

        eval_protocol = ClassificationEvalProtocol()
    else:
        eval_protocol = eval_protocol_cls()

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    model.eval()

    # Determine decision threshold on validation or training data
    opt_thresh = 0.5
    calib_loader = datamodule.val_dataloader() or datamodule.train_dataloader()
    if calib_loader:
        try:
            calib_res = eval_protocol.evaluate(model, calib_loader, device=device)
            opt_thresh = calib_res.get("opt_thresh", 0.5)
        except Exception as e:
            logger.warning("Could not compute optimal threshold on validation loader: %s", e)

    tr_idxs, _ = delegate.get_split_indices(0) if hasattr(delegate, "get_split_indices") else ([], [])
    if hasattr(delegate, "get_training_sequences"):
        seqs, input_dim = delegate.get_training_sequences()
        x_train = [seqs[i] for i in tr_idxs]
    else:
        x_train = None
        input_dim = getattr(delegate, "input_dim", 64)

    use_v = False
    if hasattr(cfg, "model") and hasattr(cfg.model, "get"):
        use_v = bool(cfg.model.get("use_v", False))

    tau_results = {
        "tau": [],
        "auc": [],
        "auc_sem": [],
        "auprc": [],
        "auprc_sem": [],
        "f1_opt": [],
        "f1_opt_sem": [],
        "f1_05": [],
        "f1_05_sem": [],
    }

    for tau in horizons:
        try:
            eval_loader = delegate.get_eval_dataloader(
                split_idx=0,
                tau=int(tau),
                x_train=x_train,
                use_v=use_v,
                input_dim=input_dim,
            )
            if eval_loader is None:
                continue
            m = eval_protocol.evaluate(model, eval_loader, device=device, opt_thresh=opt_thresh)
            tau_results["tau"].append(int(tau))
            for k in ["auc", "auprc", "f1_opt", "f1_05"]:
                tau_results[k].append(float(m.get(k, 0.0)))
                tau_results[f"{k}_sem"].append(0.0)
        except Exception as e:
            logger.warning("Horizon evaluation failed for tau=%s: %s", tau, e)

    if not tau_results["tau"]:
        logger.warning("No horizon evaluation data was collected.")
        return

    clean_exp = Path(cfg.experiment_id).stem
    plot_dir = Path("results/plots") / str(cfg.group) / clean_exp
    plot_dir.mkdir(parents=True, exist_ok=True)

    method_name = get_method_name(cfg)
    clean_key = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    json_path = plot_dir / f"metrics_{clean_key}.json"
    with open(json_path, "w") as f:
        json.dump(tau_results, f, indent=2)
    print(f"\n[Supervised Horizon Eval] Saved lead-time sweep metrics ({len(tau_results['tau'])} horizons) to: {json_path}")

    # Also save with canonical model keys for compatibility with DISP_MAP
    if clean_key.startswith("ep_"):
        alt_key = clean_key.replace("ep_", "") + ("_with_v" if use_v else "_no_v")
        alt_json_path = plot_dir / f"metrics_{alt_key}.json"
        with open(alt_json_path, "w") as f:
            json.dump(tau_results, f, indent=2)


if __name__ == "__main__":
    main()

