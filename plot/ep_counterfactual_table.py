#!/usr/bin/env python3
"""
plot/ep_counterfactual_table.py — Early Prediction Counterfactual Tables Generator.

Generates CSV tables in output_dir / "counterfactuals":
1. Metric by lead-time tables (model x tau) for all metrics from the 4-panel lead time sweeps:
   - auc_roc.csv
   - auprc.csv
   - f1_optimal.csv
   - f1_standard.csv
2. Clinical policy alignment summary table:
   - policy_alignment.csv
3. Checkpoint provenance:
   - checkpoint_info.json (records filepath of the MIMIC checkpoint used)
4. Counterfactual shock risk over lead time (if available):
   - counterfactual_shock_risk_by_tau.csv
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from plot.base import BasePlotter
from plot.ep_cache import get_ep_eval_data
from plot.ep_dl_sweep import DISP_MAP
from src.usr.methods.method_style_registry import get_style


class EpCounterfactualTablePlotter(BasePlotter):
    def __init__(self):
        super().__init__("ep_counterfactual_table")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        cf_dir = output_dir / "counterfactuals"
        cf_dir.mkdir(parents=True, exist_ok=True)

        data = get_ep_eval_data(exp_id, cfg, group, output_dir)
        checkpoint_path = None
        checkpoint_root = None
        policies = {}

        if data:
            checkpoint_path = data.get("cql_ckpt_path")
            checkpoint_root = data.get("checkpoint_root")
            policies = data.get("policies", {})

        # Fallback to config if not captured in cache
        if not checkpoint_path:
            ep_cfg = cfg.get("early_prediction", {}) if hasattr(cfg, "get") else {}
            if hasattr(ep_cfg, "get"):
                checkpoint_path = ep_cfg.get("checkpoint") or cfg.get("checkpoint")

        # 1. Record filepath of checkpoint used
        ckpt_info = {
            "cql_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_root": str(checkpoint_root) if checkpoint_root else None,
            "policies": policies or {},
        }
        with open(cf_dir / "checkpoint_info.json", "w") as f:
            json.dump(ckpt_info, f, indent=2)

        # 2. Build (model by lead time) tables for each metric from metrics_*.json
        self._write_lead_time_metric_tables(output_dir, cf_dir)

        # 3. Write policy counterfactual alignment table if cf_data is available
        if data and data.get("cf_data"):
            self._write_policy_alignment_table(data["cf_data"], cf_dir, checkpoint_path)

        # 4. Write counterfactual shock risk by tau table if ep_shock_results is available
        if data and data.get("ep_shock_results"):
            self._write_shock_risk_by_tau_table(data["ep_shock_results"], cf_dir)

        print(f"  Saved counterfactual tables in: {cf_dir}")

    def _write_lead_time_metric_tables(self, output_dir: Path, cf_dir: Path) -> None:
        """Create (model by lead time) CSV tables for AUC, AUPRC, F1_opt, F1_05."""
        metrics_dir = output_dir / "metrics"
        if metrics_dir.exists():
            json_files = list(metrics_dir.glob("metrics_*.json"))
        else:
            json_files = list(output_dir.glob("metrics_*.json"))
        if not json_files:
            json_files = list(output_dir.rglob("metrics_*.json"))

        all_models_data = {}
        all_taus = set()

        for jf in json_files:
            if jf.stat().st_size == 0:
                continue
            try:
                m_key = jf.stem.replace("metrics_", "")
                disp_name = DISP_MAP.get(m_key, m_key)
                with open(jf) as f:
                    content = json.load(f)
                if content.get("tau"):
                    all_models_data[disp_name] = content
                    all_taus.update(content["tau"])
            except Exception as e:
                print(f"Warning loading {jf} in counterfactual table plotter: {e}")

        if not all_models_data or not all_taus:
            return

        sorted_taus = sorted(list(all_taus))
        model_names = sorted(all_models_data.keys())

        metric_specs = [
            ("auc", "auc_roc.csv"),
            ("auprc", "auprc.csv"),
            ("f1_opt", "f1_optimal.csv"),
            ("f1_05", "f1_standard.csv"),
        ]

        tau_cols = [f"tau_{t}" for t in sorted_taus]

        for metric_key, filename in metric_specs:
            rows = []
            for m_name in model_names:
                m_data = all_models_data[m_name]
                tau_map = {t: val for t, val in zip(m_data["tau"], m_data.get(metric_key, []))}
                row = {"model": m_name}
                for t, col in zip(sorted_taus, tau_cols):
                    val = tau_map.get(t, None)
                    row[col] = round(float(val), 4) if val is not None else None
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = cf_dir / filename
            df.to_csv(csv_path, index=False)

    def _write_policy_alignment_table(self, cf_data: list, cf_dir: Path, checkpoint_path: Optional[str]) -> None:
        """Write clinician vs policy counterfactual alignment summary table."""
        rows = []
        for r in cf_data:
            rows.append({
                "method": r["method"],
                "checkpoint": checkpoint_path if r["method"] != "clinician" else "Dataset Clinician",
                "accuracy_agr_pct_mean": round(r.get("agreement_mean", 0.0) * 100, 2),
                "accuracy_agr_pct_sem": round(r.get("agreement_sem", 0.0) * 100, 2),
                "admin_rate_pct_mean": round(r.get("admin_rate_mean", 0.0) * 100, 2),
                "admin_rate_pct_sem": round(r.get("admin_rate_sem", 0.0) * 100, 2),
                "precision_mean": round(r.get("precision_mean", 0.0), 4),
                "precision_sem": round(r.get("precision_sem", 0.0), 4),
                "recall_mean": round(r.get("recall_mean", 0.0), 4),
                "recall_sem": round(r.get("recall_sem", 0.0), 4),
                "f1_score_mean": round(r.get("f1_mean", 0.0), 4),
                "f1_score_sem": round(r.get("f1_sem", 0.0), 4),
                "pred_mortality_pct_mean": round(r.get("pred_mort_mean", 0.0) * 100, 2),
                "pred_mortality_pct_sem": round(r.get("pred_mort_sem", 0.0) * 100, 2),
            })
        df = pd.DataFrame(rows)
        df.to_csv(cf_dir / "policy_alignment.csv", index=False)

    def _write_shock_risk_by_tau_table(self, ep_shock_results: dict, cf_dir: Path) -> None:
        """Write counterfactual shock risk predicted by lead time tau."""
        rows = []
        for policy_name, p_data in ep_shock_results.items():
            taus = p_data.get("tau", [])
            means = p_data.get("all", {}).get("means", [])
            row = {"policy": policy_name}
            for t, m in zip(taus, means):
                row[f"tau_{t}"] = round(float(m), 4)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(cf_dir / "counterfactual_shock_risk_by_tau.csv", index=False)
