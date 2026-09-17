import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import logging
import hydra
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import omegaconf

logger = logging.getLogger(__name__)

try:
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.Container,
        omegaconf.nodes.UntypedNode,
    ])
except (AttributeError, TypeError) as e:
    logger.debug("PyTorch safe globals registration skipped: %s", e)
except Exception as e:
    logger.debug("Unexpected error registering safe globals: %s", e)

from src.core.lightning_builder import build_trainer, finalize_training
from src.methods.registry import auto_discover, get_agent_class
from src.data.rl_data_module import RLDataModule

@hydra.main(version_base=None, config_path="../in/config", config_name="config")
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


    auto_discover()
    
    agent_cfg = cfg.agent
    base_algo_name = agent_cfg.get("algorithm", agent_cfg.get("name", None))
    print(f"Extracted algorithm name: {base_algo_name}")

    if not base_algo_name:
        raise ValueError("Could not extract algorithm name from config.")

    AgentClass = get_agent_class(base_algo_name)
    print(f"Resolved agent class: {AgentClass.__name__}")
    model = AgentClass(cfg)
    
    print("\n" + "="*30)
    print("      AGENT ARCHITECTURE")
    print("="*30)
    print(model)
    
    if hasattr(model, "model") and hasattr(model.model, "_print"):
        print("\n" + "="*30)
        print("      LOGIC RULES & BLENDER")
        print("="*30)
        model.model._print()
    print("="*30 + "\n")
    
    datamodule = RLDataModule(cfg)
    
    trainer, ckpt_dir, ckpt_path = build_trainer(cfg, model)

    start_time = time.time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n[Training Complete] Total execution time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    metric = finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time)
    return metric

if __name__ == "__main__":
    main()
