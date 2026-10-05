import os
import pickle
from pathlib import Path


def get_ep_eval_data(exp_id, cfg, group, output_dir):
    cache_path = Path(output_dir) / "ep_eval_cache.pkl"
    remake = cfg.get("remake", False)

    from plot.base import BasePlotter
    dummy = BasePlotter("ep_cache")
    root_cfg = dummy.get_experiment_config(exp_id)
    ep_cfg = root_cfg.get("early_prediction", {}) if isinstance(root_cfg.get("early_prediction"), dict) else {}

    explicit_ckpt = (
        cfg.get("checkpoint")
        or root_cfg.get("checkpoint")
        or ep_cfg.get("checkpoint")
    )
    if not explicit_ckpt:
        return None

    ckpt_dir = Path(explicit_ckpt)
    if not ckpt_dir.exists():
        return None

    cache_stale = False
    if cache_path.exists() and ckpt_dir.exists():
        cache_mtime = cache_path.stat().st_mtime
        for ckpt_file in ckpt_dir.rglob("*.ckpt"):
            if ckpt_file.stat().st_mtime > cache_mtime:
                cache_stale = True
                break

    if cache_path.exists() and not remake and not cache_stale:
        print(f"Loading cached EP evaluation data from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from src.usr.eval.early_prediction.eval_logic import compute_ep_eval_data

    if not ckpt_dir.exists():
        print(f"Error: Could not find checkpoint directory for {exp_id}")
        return None

    print("Computing EP evaluation data (this may take a while)...")
    data = compute_ep_eval_data(
        checkpoint_root=str(ckpt_dir),
        dataset_path=cfg.get("dataset_path") or ep_cfg.get("dataset_path", None),
        ep_ckpt_root=cfg.get("ep_ckpt_root") or ep_cfg.get("ep_ckpt_root", "results/checkpoints/early_prediction"),
        n_splits=cfg.get("n_splits") or ep_cfg.get("n_splits", 20),
        use_volatility=cfg.get("use_volatility", ep_cfg.get("use_volatility", True)),
    )

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data
