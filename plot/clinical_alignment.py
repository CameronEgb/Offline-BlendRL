#!/usr/bin/env python3
"""
plot/clinical_alignment.py — MIMIC Clinical Policy Alignment & Septic Shock Plotter.

Evaluates trained RL policies against clinician treatment decisions and patient
septic shock outcomes in ICU time-series trajectories.

Outputs:
  - Clinical action alignment: precision, recall, F1, windowed F1, AUC-ROC, AUPRC
  - Septic shock outcome analysis across clinician agreement deciles
  - Visual figures: clinician_agreement_vs_shock.png, clinical_agreement.png (opt-in)
  - Tabular metrics: method_comparison.csv
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from plot.base import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.method_registry import get_style as get_method_style


class ClinicalAlignmentPlotter(BasePlotter):
    def __init__(self):
        super().__init__("clinical_alignment")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        self._run_mimic_eval(exp_id, cfg, group, clean_exp, output_dir)

    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str):
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases = set()
        has_active_filter = False
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, [])
            if val:
                has_active_filter = True
                if isinstance(val, (list, tuple)):
                    methods = list(val)
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    active_aliases.update(get_method_aliases(m))

        method_ckpts = {}
        for method_dir in sorted(ckpt_root.iterdir()):
            if method_dir.is_dir():
                m_name = method_dir.name
                if has_active_filter and m_name not in active_aliases:
                    continue

                best_ckpt = None
                storage_url = exp_cfg.get("hydra", {}).get("sweeper", {}).get("storage", None)
                if storage_url:
                    from src.pipeline.optuna_utils import get_best_trial_id
                    study_name = f"{clean_exp}_{m_name}"
                    best_id = get_best_trial_id(storage_url, study_name)
                    candidate = method_dir / best_id / "best_model.ckpt"
                    if candidate.exists():
                        best_ckpt = candidate

                if not best_ckpt:
                    ckpts = list(method_dir.rglob("best_model*.ckpt"))
                    if ckpts:
                        best_ckpt = ckpts[0]

                if best_ckpt:
                    canon = get_canonical_method_name(m_name)
                    if canon not in method_ckpts or m_name == canon:
                        method_ckpts[canon] = best_ckpt
        return method_ckpts

    def _load_agent(self, path, dev):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        last_error = None
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                ag.to(dev)
                ag.eval()
                return ag
            except Exception as e:
                last_error = e
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False, strict=False)
                    ag.to(dev)
                    ag.eval()
                    return ag
                except Exception as e2:
                    last_error = e2
                    continue
        if last_error is not None:
            print(f"  [clinical_alignment] Checkpoint load error for {path}: {last_error}")
        return None

    def _get_probs_and_actions(self, ag, obs_b):
        if hasattr(ag, "get_action_probs"):
            probs = ag.get_action_probs(obs_b)
            acts = ag.get_action(obs_b) if hasattr(ag, "get_action") else torch.argmax(probs, dim=-1)
            return probs, acts

        # Fallback for unmigrated or raw models
        is_cql = ag.__class__.__name__ == "CQLAgent" or "cql" in str(getattr(ag, "algorithm", "")).lower()
        use_actor = bool(ag.get_cfg("use_actor", False)) if hasattr(ag, "get_cfg") else getattr(ag, "use_actor", False)

        if hasattr(ag, "is_modular") and ag.is_modular:
            logic_obs = ag._prepare_logic_obs(obs_b) if hasattr(ag, "_prepare_logic_obs") else obs_b.unsqueeze(1).repeat(1, 2, 1)
            if is_cql and not use_actor and hasattr(ag.model, "get_q_values"):
                q_vals = ag.model.get_q_values(obs_b, logic_obs)
                probs = torch.softmax(q_vals, dim=-1)
                acts = torch.argmax(q_vals, dim=-1)
                return probs, acts
            elif hasattr(ag.model, "actor"):
                probs, _ = ag.model.actor(obs_b, logic_obs)
                acts = torch.argmax(probs, dim=-1)
                return probs, acts
            elif hasattr(ag.model, "get_q_values"):
                q_vals = ag.model.get_q_values(obs_b, logic_obs)
                probs = torch.softmax(q_vals, dim=-1)
                acts = torch.argmax(q_vals, dim=-1)
                return probs, acts
        elif is_cql and not use_actor and hasattr(ag, "q_network"):
            q = ag.q_network.get_q_values(obs_b) if hasattr(ag.q_network, "get_q_values") else ag.q_network(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "actor") and hasattr(ag.actor, "get_action_probs"):
            probs = ag.actor.get_action_probs(obs_b)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "fuzzy_model") and ag.fuzzy_model is not None:
            q = ag.fuzzy_model(obs_b.to("cpu"))
            probs = torch.softmax(q, dim=-1).to(obs_b.device)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "q_network"):
            if hasattr(ag.q_network, "get_action_probs"):
                probs = ag.q_network.get_action_probs(obs_b)
            else:
                q = ag.q_network(obs_b)
                probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
            q = ag.model.get_q_values(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        else:
            out = ag.get_action_and_value(obs_b)
            act = out[0] if isinstance(out, (tuple, list)) else out
            n_acts = 3 if obs_b.shape[-1] >= 123 else 2
            probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
            probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
            return probs, act

    def _run_mimic_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        """Clinical policy alignment and septic shock evaluation for MIMIC datasets."""
        env_ds = cfg.get("env", {}).get("dataset_name", "mimic_lazy_0_interventions_balanced.npz") if isinstance(cfg.get("env"), dict) else "mimic_lazy_0_interventions_balanced.npz"
        npz_candidate = Path("in/datasets/mimic") / env_ds
        if not npz_candidate.exists():
            mode_path = cfg.get("mode", {}).get("dataset_path", "")
            if mode_path:
                cand = Path(mode_path).with_suffix(".npz")
                if cand.exists():
                    npz_candidate = cand

        if not npz_candidate.exists():
            raise FileNotFoundError(f"[clinical_alignment]: MIMIC dataset file '{npz_candidate}' not found.")

        print(f"\n==========================================================================================")
        print(f"=== Running MIMIC Clinical Alignment Evaluation for '{exp_id}' ===")
        print(f"==========================================================================================")
        data = np.load(npz_candidate, allow_pickle=True)
        X = data['X']        # (N, 240, 49)
        mask = data['mask']  # (N, 240, 1)

        valid_mask = (mask.squeeze(-1) != -1)
        all_obs = X[:, :, :46][valid_mask]
        all_clin_acts = X[:, :, 47][valid_mask].astype(int)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        total_steps = len(all_clin_acts)

        cache_path = output_dir / "clinical_alignment_cache.npz"
        remake = cfg.get("remake", False)
        num_patients = X.shape[0]
        outcomes = data['y'].squeeze() if 'y' in data else np.zeros(num_patients)
        patient_agreements = {}

        method_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not method_ckpts and cache_path.exists() and not remake:
            print(f"  Loading cached clinical alignment data from {cache_path}")
            cached_data = np.load(cache_path, allow_pickle=True)
            for k in cached_data.files:
                if k != "outcomes":
                    patient_agreements[k] = cached_data[k]
            if "outcomes" in cached_data.files:
                outcomes = cached_data["outcomes"]
        elif not method_ckpts:
            print(f"Notice [clinical_alignment]: No policy checkpoints found for '{clean_exp}'")
            return

        results = []

        # 1. Clinician Baseline
        clin_admin_rate = (all_clin_acts == 1).mean() * 100.0
        results.append({
            "Method": "Clinician (Baseline)",
            "Accuracy %": 100.0,
            "Admin Rate %": float(clin_admin_rate),
            "AUC-ROC": 1.0000,
            "AUPRC": 1.0000,
            "Precision": 1.0000,
            "Recall": 1.0000,
            "F1 Score": 1.0000,
            "Best F1": 1.0000,
            "Windowed F1 (±3h)": 1.0000,
            "Windowed Recall %": 100.0,
            "Opt Threshold": 0.5000
        })

        batch_size = 10000
        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                print(f"  Warning [clinical_alignment]: Could not load checkpoint {ckpt_path}")
                continue

            all_admin_probs = []
            all_policy_acts = []

            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_batch = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)

                    probs, policy_acts_tensor = self._get_probs_and_actions(agent, obs_batch)
                    policy_acts = policy_acts_tensor.cpu().numpy()
                    if probs.shape[-1] > 1:
                        admin_probs = probs[:, 1].cpu().numpy()
                    else:
                        admin_probs = probs.squeeze().cpu().numpy()

                    all_admin_probs.extend(admin_probs)
                    all_policy_acts.extend(policy_acts)

            all_admin_probs = np.array(all_admin_probs)
            all_policy_acts = np.array(all_policy_acts)

            matches = (all_policy_acts == all_clin_acts).sum()
            accuracy = (matches / total_steps) * 100.0
            admin_rate = (all_policy_acts == 1).mean() * 100.0

            tp = ((all_policy_acts == 1) & (all_clin_acts == 1)).sum()
            fp = ((all_policy_acts == 1) & (all_clin_acts == 0)).sum()
            fn = ((all_policy_acts == 0) & (all_clin_acts == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            try:
                auc_roc = float(roc_auc_score(all_clin_acts, all_admin_probs))
            except Exception:
                auc_roc = float('nan')

            try:
                auprc = float(average_precision_score(all_clin_acts, all_admin_probs))
            except Exception:
                auprc = float('nan')

            # Best F1 threshold sweep
            try:
                p_thresh, r_thresh, thresholds = precision_recall_curve(all_clin_acts, all_admin_probs)
                f1_scores = 2 * (p_thresh * r_thresh) / (p_thresh + r_thresh + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_f1 = float(f1_scores[best_idx])
                opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            except Exception:
                best_f1 = f1
                opt_thresh = 0.5

            # Windowed agreement calculation (±3 hours)
            step_idx = 0
            patient_agrs = []
            win_tp = 0
            win_fp = 0
            win_fn = 0
            total_clin_pos = 0

            for p_idx in range(num_patients):
                p_valid = valid_mask[p_idx]
                p_len = p_valid.sum()
                if p_len == 0:
                    continue

                p_clin = all_clin_acts[step_idx:step_idx + p_len]
                p_pol = all_policy_acts[step_idx:step_idx + p_len]
                step_idx += p_len

                p_agree = (p_clin == p_pol).mean()
                patient_agrs.append(p_agree)

                clin_pos_indices = np.where(p_clin == 1)[0]
                pol_pos_indices = np.where(p_pol == 1)[0]
                total_clin_pos += len(clin_pos_indices)

                for c_pos in clin_pos_indices:
                    if len(pol_pos_indices) > 0 and np.min(np.abs(pol_pos_indices - c_pos)) <= 3:
                        win_tp += 1
                    else:
                        win_fn += 1

                for p_pos in pol_pos_indices:
                    if len(clin_pos_indices) == 0 or np.min(np.abs(clin_pos_indices - p_pos)) > 3:
                        win_fp += 1

            patient_agreements[method_name] = np.array(patient_agrs)
            win_precision = win_tp / (win_tp + win_fp + 1e-8)
            win_recall = win_tp / (total_clin_pos + 1e-8)
            win_f1 = 2 * (win_precision * win_recall) / (win_precision + win_recall + 1e-8)

            results.append({
                "Method": clean_label(method_name),
                "Accuracy %": float(accuracy),
                "Admin Rate %": float(admin_rate),
                "AUC-ROC": auc_roc,
                "AUPRC": auprc,
                "Precision": float(precision),
                "Recall": float(recall),
                "F1 Score": float(f1),
                "Best F1": best_f1,
                "Windowed F1 (±3h)": float(win_f1),
                "Windowed Recall %": float(win_recall * 100.0),
                "Opt Threshold": opt_thresh
            })

        if method_ckpts:
            df = pd.DataFrame(results)
            csv_path = output_dir / "method_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved MIMIC method comparison: {csv_path}")
            np.savez(cache_path, outcomes=outcomes, **patient_agreements)
            print(f"  Cached clinical alignment data: {cache_path}")

        # Color deduplication across active methods
        color_map = self._deduplicate_colors(list(patient_agreements.keys()))

        # 1. Clinician Agreement % Bar Chart (Opt-in only, excluded from defaults)
        requested_plots = cfg.get("plots", [])
        if isinstance(requested_plots, dict):
            requested_plots = list(requested_plots.keys())
        elif isinstance(requested_plots, (list, tuple)):
            requested_plots = [str(p).lower() for p in requested_plots]
        else:
            requested_plots = []

        if any(k in requested_plots for k in ["clinical_agreement", "agreement_bar", "agreement"]) and results:
            fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 5.5))
            methods = [r["Method"] for r in results]
            accuracies = [r["Accuracy %"] for r in results]

            bar_colors = []
            for r in results:
                m_name = r["Method"]
                if "Clinician" in m_name:
                    bar_colors.append("#756bb1")
                else:
                    bar_colors.append(color_map.get(m_name, get_method_style(m_name).get("color") or "tab:blue"))

            bars = ax.bar(methods, accuracies, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=1.0, alpha=0.85)
            ax.set_ylabel("Clinician Agreement (%)", fontsize=12, fontweight="bold")
            ax.set_title(f"MIMIC Treatment Action Agreement ({clean_exp})", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="bold")

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

            fig.tight_layout()
            plot_path = output_dir / "clinical_agreement.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {plot_path}")

        # 2. Clinician Agreement vs Septic Shock Outcome Analysis
        plot_all = len(requested_plots) == 0
        should_plot_abs = plot_all or any(k in requested_plots for k in ["agreement_vs_shock", "agreement_vs_shock_absolute", "shock", "absolute"])
        should_plot_dec = plot_all or any(k in requested_plots for k in ["agreement_vs_shock_deciles", "deciles", "shock_deciles"])

        has_valid_data = patient_agreements and len(outcomes) == len(next(iter(patient_agreements.values())))

        if should_plot_abs and has_valid_data:
            self._plot_agreement_vs_shock_absolute(patient_agreements, outcomes, output_dir, clean_exp, color_map)

        if should_plot_dec and has_valid_data:
            self._plot_agreement_vs_shock_deciles(patient_agreements, outcomes, output_dir, clean_exp, color_map)

        print("==========================================================================================\n")

    def _deduplicate_colors(self, methods: List[str]) -> Dict[str, str]:
        """Ensures every method plotted has a distinct color, dynamically reassigning duplicates if needed."""
        fallback_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#08519c", "#d95f02", "#018571", "#6a3d9a", "#e7298a"
        ]
        color_map = {}
        used_colors = set()
        for m in methods:
            style = get_method_style(m)
            c = style.get("color")
            if not c or c in used_colors:
                for candidate in fallback_palette:
                    if candidate not in used_colors:
                        c = candidate
                        break
                else:
                    c = f"C{len(color_map) % 10}"
            color_map[m] = c
            used_colors.add(c)
        return color_map

    def _plot_agreement_vs_shock_deciles(self, patient_agreements: dict, outcomes: np.ndarray,
                                         output_dir: Path, clean_exp: str, color_map: dict):
        """Plot septic shock incidence across 10 equal patient agreement deciles with background distribution."""
        fig, ax = plt.subplots(figsize=(11, 6))
        ax2 = ax.twinx()

        num_patients = len(outcomes)
        decile_labels = [f"D{i+1}\n({i*10}-{(i+1)*10}%)" for i in range(10)]
        x_indices = np.arange(10)

        # Background distribution: in deciles, each bucket has ~N/10 patients
        patients_per_decile = [num_patients // 10] * 10
        patients_per_decile[-1] += num_patients % 10

        ax2.bar(x_indices, patients_per_decile, width=0.55, color="#cfd8dc", edgecolor="#90a4ae",
                alpha=0.40, linewidth=1.0, label="Trajectories per Decile", zorder=1)
        ax2.set_ylabel("Number of Trajectories", fontsize=11, fontweight="bold", color="#546e7a")
        ax2.tick_params(axis='y', labelcolor="#546e7a")
        ax2.set_ylim(0, max(patients_per_decile) * 1.55)

        # Foreground lines
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        for m_name, p_agr in sorted(patient_agreements.items()):
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
            marker = get_method_style(m_name).get("marker") or "o"
            ls = get_method_style(m_name).get("linestyle") or "-"

            # Equal frequency rank deciles (0 to 9)
            ranks = pd.Series(p_agr).rank(method="first").values
            decile_assignments = pd.qcut(ranks, q=10, labels=False)

            shock_rates = []
            for d in range(10):
                m = (decile_assignments == d)
                shock_rates.append(outcomes[m].mean() * 100.0)

            ax.plot(x_indices, shock_rates, marker=marker, linestyle=ls, linewidth=2.2, markersize=7,
                    label=clean_label(m_name), color=color, zorder=5)

        ax.set_xlabel("Patient Agreement Decile (Lowest → Highest Clinician Alignment)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Septic Shock Incidence (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"Septic Shock Incidence vs Clinician Agreement Deciles ({clean_exp})", fontsize=13, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(decile_labels, fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_ylim(0, 105)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.08, 1),
                  fontsize=9.5, framealpha=0.95)

        fig.tight_layout()
        decile_plot_path = output_dir / "clinician_agreement_vs_shock_deciles.png"
        plt.savefig(decile_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Decile Shock Plot:   {decile_plot_path}")

    def _plot_agreement_vs_shock_absolute(self, patient_agreements: dict, outcomes: np.ndarray,
                                          output_dir: Path, clean_exp: str, color_map: dict):
        """Plot septic shock incidence across standardized absolute clinician agreement intervals."""
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(11, 6))
        ax2 = ax.twinx()

        bin_edges = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
        bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(10)]
        x_indices = np.arange(10)
        n_methods = len(patient_agreements)
        bar_width = 0.8 / max(1, n_methods)

        max_count = 0
        method_items = sorted(patient_agreements.items())

        # Plot background trajectory distribution grouped bars for each method
        for idx, (m_name, p_agr) in enumerate(method_items):
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
            counts = []
            for i in range(10):
                low = bin_edges[i]
                high = bin_edges[i+1]
                m = (p_agr >= low) & (p_agr <= high if i == 9 else p_agr < high)
                counts.append(m.sum())
            max_count = max(max_count, max(counts) if counts else 0)

            x_pos = x_indices + (idx - (n_methods - 1) / 2.0) * bar_width
            ax2.bar(x_pos, counts, width=bar_width * 0.9, color=color, alpha=0.20,
                    edgecolor=color, linewidth=0.8, zorder=1)

        ax2.set_ylabel("Number of Trajectories in Bucket", fontsize=11, fontweight="bold", color="#546e7a")
        ax2.tick_params(axis='y', labelcolor="#546e7a")
        ax2.set_ylim(0, max(1, max_count) * 1.35)

        # Foreground shock incidence lines
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        for m_name, p_agr in method_items:
            color = color_map.get(m_name) or get_method_style(m_name).get("color") or "tab:blue"
            marker = get_method_style(m_name).get("marker") or "o"
            ls = get_method_style(m_name).get("linestyle") or "-"

            shock_rates = []
            for i in range(10):
                low = bin_edges[i]
                high = bin_edges[i+1]
                m = (p_agr >= low) & (p_agr <= high if i == 9 else p_agr < high)
                if m.sum() >= 5:  # Require at least 5 patients to plot reliable point
                    shock_rates.append(outcomes[m].mean() * 100.0)
                else:
                    shock_rates.append(np.nan)

            valid_mask = ~np.isnan(shock_rates)
            if valid_mask.sum() > 0:
                ax.plot(x_indices[valid_mask], np.array(shock_rates)[valid_mask],
                        marker=marker, linestyle=ls, linewidth=2.2, markersize=7,
                        label=clean_label(m_name), color=color, zorder=5)

        ax.set_xlabel("Clinician Agreement Rate Interval (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Septic Shock Incidence (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"Septic Shock Incidence vs Absolute Clinician Agreement ({clean_exp})", fontsize=13, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(bin_labels, fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_ylim(0, 105)

        lines1, labels1 = ax.get_legend_handles_labels()
        dist_patch = Patch(facecolor="#90a4ae", alpha=0.35, label="Trajectory Distribution (Bars)")
        ax.legend(lines1 + [dist_patch], labels1 + ["Trajectory Distribution (Bars)"],
                  loc="upper left", bbox_to_anchor=(1.08, 1), fontsize=9.5, framealpha=0.95)

        fig.tight_layout()
        abs_plot_path = output_dir / "clinician_agreement_vs_shock_absolute.png"
        plt.savefig(abs_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Absolute Shock Plot: {abs_plot_path}")


