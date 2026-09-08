#!/usr/bin/env python3
"""
plot/blend_routing.py — BlendRL Neuro-Symbolic Routing, Authority & OOD Plotter.

Visualizes how the BlendRL gating network (blender) distributes decision-making authority
between symbolic logic and neural policies across:
  1. Dataset-Wide Routing Authority Breakdown (Stacked Bar: % Pure Logic, % Pure Neural, % Mixed)
  2. Routing Authority by Student Competency Tier (Grouped Bar across Low, Med, High Tiers)
  3. Multi-View State-Space Decision Boundaries (Simplex, Full PCA, and t-SNE Manifold)
  4. Out-of-Distribution (OOD) Manifold & Balance of Expertise Handover Curves
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from plot.base import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.pyrenees_evaluator import PyreneesEvaluator


class BlendRoutingPlotter(BasePlotter):
    def __init__(self):
        super().__init__("blend_routing")
        self.tier_names = [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        self._run_blend_routing_eval(exp_id, cfg, group, clean_exp, output_dir)

    def _discover_blendrl_checkpoints(self, exp_id: str, group: str, clean_exp: str) -> dict:
        """Discovers only modular BlendRL checkpoints that output blending weights."""
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        known_problems = ["problem", "ex132(w)", "ex132a(w)", "ex152a(w)", "ex212(w)", "ex242(w)", "ex252(w)", "ex252a(w)", "exc137(w)", "exp426d(w)", "exp426e(w)"]
        discovered = {}

        for entry in sorted(ckpt_root.rglob("best_model*.ckpt")):
            rel_parts = entry.relative_to(ckpt_root).parts
            parent_dir_name = rel_parts[0]

            # Only target BlendRL modular agents (skip pure neural CQL/DNN/PPO)
            if "blendrl" not in parent_dir_name.lower():
                continue

            detected_dataset = None
            detected_method = parent_dir_name

            if parent_dir_name in known_problems and len(rel_parts) > 2:
                detected_dataset = parent_dir_name
                detected_method = rel_parts[1]
            else:
                for prob in known_problems:
                    prob_clean = prob.replace("(", "_").replace(")", "_").rstrip("_")
                    if parent_dir_name.endswith(f"_{prob}"):
                        detected_dataset = prob
                        detected_method = parent_dir_name[:-len(f"_{prob}")]
                        break
                    elif parent_dir_name.endswith(f"_{prob_clean}"):
                        detected_dataset = prob
                        detected_method = parent_dir_name[:-len(f"_{prob_clean}")].rstrip("_")
                        break

            canon_method = get_canonical_method_name(detected_method)
            key = f"{canon_method}_{detected_dataset}" if detected_dataset else canon_method
            if key not in discovered or entry.name == "best_model.ckpt":
                discovered[key] = {
                    "path": entry,
                    "method": canon_method,
                    "dataset": detected_dataset or "problem",
                    "dir_name": parent_dir_name,
                }
        return discovered

    def _load_agent(self, path: Path):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location="cpu", weights_only=False)
                ag.eval()
                if hasattr(ag, "is_modular") and ag.is_modular:
                    return ag
            except Exception:
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location="cpu", weights_only=False, strict=False)
                    ag.eval()
                    if hasattr(ag, "is_modular") and ag.is_modular:
                        return ag
                except Exception:
                    continue
        return None

    def _run_blend_routing_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        print(f"\n==========================================================================================")
        print(f"=== BlendRL Routing & Authority Evaluation ({clean_exp}) ===")
        print(f"==========================================================================================")

        discovered = self._discover_blendrl_checkpoints(exp_id, group, clean_exp)
        if not discovered:
            print(f"Notice [blend_routing]: No modular BlendRL checkpoints found for '{clean_exp}'")
            return

        print(f"Discovered {len(discovered)} BlendRL model checkpoints to evaluate.")

        evaluator = PyreneesEvaluator()
        authority_rows = []
        tier_rows = []
        ood_handover_rows = []
        state_space_data = {}

        for key, meta in sorted(discovered.items()):
            prob_name = meta["dataset"]
            agent = self._load_agent(meta["path"])
            if agent is None:
                continue

            # Load dataset states
            clean_path = Path(f"in/datasets/pyrenees/per_problem/{prob_name}/clean.npz")
            if not clean_path.exists() and prob_name == "problem":
                clean_path = Path("in/datasets/pyrenees/pyrenees_clean.npz")
            if not clean_path.exists():
                continue

            data = np.load(clean_path, allow_pickle=True)
            states = np.vstack(data["states"]).astype(np.float32)
            n_states = len(states)
            if n_states == 0:
                continue

            # Compute blending weights
            obs_t = torch.tensor(states, dtype=torch.float32)
            pad = torch.zeros((len(obs_t), 3), dtype=torch.float32)
            obs_aug = torch.cat([obs_t, pad], dim=-1) if obs_t.shape[-1] == 123 else obs_t
            logic_obs = obs_aug.unsqueeze(1).repeat(1, 2, 1)

            with torch.no_grad():
                probs, weights = agent.model.actor(obs_aug, logic_obs)

            # Dynamically look up logic and neural indices from the model's module_types
            module_types = getattr(agent.model.actor, "module_types", ["logic", "neural"])
            logic_idx = module_types.index("logic") if "logic" in module_types else 0
            neural_idx = module_types.index("neural") if "neural" in module_types else (1 if len(module_types) > 1 else 0)

            w_logic = weights[:, logic_idx].numpy()
            w_neural = weights[:, neural_idx].numpy()

            # GMM Tiers
            gmm_path = Path(f"in/datasets/pyrenees/per_problem/{prob_name}/gmm_scaler.npz")
            if not gmm_path.exists():
                gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")
            tiers = evaluator._compute_gmm_tiers(states, gmm_path)

            # 1. Authority Breakdown
            pure_log_pct = float(np.mean(w_logic >= 0.90) * 100.0)
            pure_neu_pct = float(np.mean(w_neural >= 0.90) * 100.0)
            mixed_pct = float(np.mean((w_neural > 0.10) & (w_neural < 0.90)) * 100.0)

            auth_row = {
                "Model": prob_name,
                "Method": clean_label(meta["method"]),
                "N_States": n_states,
                "Mean Logic Weight": float(np.mean(w_logic)),
                "Mean Neural Weight": float(np.mean(w_neural)),
                "Pure Logic % (>=0.90)": pure_log_pct,
                "Pure Neural % (>=0.90)": pure_neu_pct,
                "Mixed % (0.10-0.90)": mixed_pct,
            }
            authority_rows.append(auth_row)

            # 2. Tier Breakdown
            for t_label, t_val in self.tier_names:
                m = (tiers == t_val)
                n_t = int(m.sum())
                if n_t > 0:
                    tier_rows.append({
                        "Model": prob_name,
                        "Method": clean_label(meta["method"]),
                        "Tier": t_label,
                        "N_States": n_t,
                        "Mean Logic Weight": float(np.mean(w_logic[m])),
                        "Mean Neural Weight": float(np.mean(w_neural[m])),
                        "Pure Logic % (>=0.90)": float(np.mean(w_logic[m] >= 0.90) * 100.0),
                        "Pure Neural % (>=0.90)": float(np.mean(w_neural[m] >= 0.90) * 100.0),
                    })

            # 3. OOD Distance & Handover Computation
            sub_n = min(n_states, 4000)
            sub_idx = np.random.default_rng(42).choice(n_states, size=sub_n, replace=False)
            sub_states = states[sub_idx]
            sub_w_logic = w_logic[sub_idx]
            sub_w_neural = w_neural[sub_idx]

            nbrs = NearestNeighbors(n_neighbors=20, metric="euclidean").fit(sub_states)
            distances, _ = nbrs.kneighbors(sub_states)
            ood_score = distances[:, -1]

            # Compute decile bins
            n_bins = 5
            bin_labels = [f"Q{i+1}" for i in range(n_bins)]
            try:
                ood_bins = pd.qcut(ood_score, q=n_bins, labels=bin_labels)
                for b_name in bin_labels:
                    m_b = (ood_bins == b_name)
                    if m_b.sum() > 0:
                        ood_handover_rows.append({
                            "Model": prob_name,
                            "Method": clean_label(meta["method"]),
                            "OOD_Quantile": b_name,
                            "Mean_OOD_Distance": float(ood_score[m_b].mean()),
                            "Mean_Logic_Weight": float(sub_w_logic[m_b].mean()),
                            "Mean_Neural_Weight": float(sub_w_neural[m_b].mean()),
                            "Pure_Logic_Pct": float((sub_w_logic[m_b] >= 0.90).mean() * 100.0),
                            "Pure_Neural_Pct": float((sub_w_neural[m_b] >= 0.90).mean() * 100.0),
                        })
            except Exception:
                pass

            # Store sample for 2D state space & manifold visualization
            if prob_name in ["problem", "ex152a(w)", "ex252a(w)", "exp426e(w)"] or len(state_space_data) == 0:
                state_space_data[prob_name] = {
                    "states": sub_states,
                    "w_logic": sub_w_logic,
                    "w_neural": sub_w_neural,
                    "tiers": tiers[sub_idx],
                    "ood_score": ood_score,
                }

        if not authority_rows:
            print("Notice [blend_routing]: No evaluatable BlendRL models.")
            return

        # Save CSVs & Markdown Report
        self._save_reports(output_dir, authority_rows, tier_rows, ood_handover_rows, clean_exp, group)

        # Generate Plots
        self._plot_routing_authority(authority_rows, output_dir, clean_exp)
        self._plot_routing_by_tier(tier_rows, output_dir, clean_exp)
        self._plot_state_space_boundaries(state_space_data, output_dir, clean_exp)
        self._plot_ood_manifold(state_space_data, output_dir, clean_exp)
        self._plot_ood_handover_curve(ood_handover_rows, output_dir, clean_exp)

        print("==========================================================================================\n")

    def _save_reports(self, output_dir: Path, auth_rows: list, tier_rows: list, ood_rows: list, clean_exp: str, group: str):
        df_auth = pd.DataFrame(auth_rows)
        auth_csv = output_dir / "blend_routing_authority.csv"
        df_auth.to_csv(auth_csv, index=False)
        print(f"  Saved Authority CSV:    {auth_csv}")

        df_tier = pd.DataFrame(tier_rows)
        tier_csv = output_dir / "blend_routing_by_tier.csv"
        df_tier.to_csv(tier_csv, index=False)
        print(f"  Saved Tier CSV:         {tier_csv}")

        if ood_rows:
            df_ood = pd.DataFrame(ood_rows)
            ood_csv = output_dir / "blend_routing_ood_handover.csv"
            df_ood.to_csv(ood_csv, index=False)
            print(f"  Saved OOD Handover CSV: {ood_csv}")

        md_path = output_dir / "blend_routing_report.md"
        with open(md_path, "w") as f:
            f.write(f"# BlendRL Neuro-Symbolic Routing & Authority Report\n\n")
            f.write(f"**Experiment**: `{clean_exp}` (Group: `{group}`)\n\n")
            f.write(f"This report details how the BlendRL gating network distributes decision-making authority between the Symbolic Logic Reasoner and the Neural Policy across in-distribution and out-of-distribution (OOD) regimes.\n\n")
            f.write("## 1. Dataset-Wide Routing Authority\n\n")
            f.write(df_auth.to_markdown(index=False))
            f.write("\n\n## 2. Authority Distribution by Student Competency Tier\n\n")
            f.write(df_tier.to_markdown(index=False))
            if ood_rows:
                f.write("\n\n## 3. Balance of Expertise across OOD Distance Quantiles\n\n")
                f.write(df_ood.to_markdown(index=False))
            f.write("\n\n---\n*Auto-generated by NeSyRL Pipeline*\n")
        print(f"  Saved Markdown Report:  {md_path}")

    def _plot_routing_authority(self, auth_rows: list, output_dir: Path, clean_exp: str):
        """Plot 1: Stacked Bar Chart of Pure Logic vs Pure Neural vs Mixed."""
        df = pd.DataFrame(auth_rows)
        models = df["Model"].tolist()
        y_pos = np.arange(len(models))

        logic_pcts = df["Pure Logic % (>=0.90)"].values
        mixed_pcts = df["Mixed % (0.10-0.90)"].values
        neural_pcts = df["Pure Neural % (>=0.90)"].values

        fig, ax = plt.subplots(figsize=(12, max(5.5, len(models) * 0.45)))

        c_logic = "#d95f02"   # Orange
        c_mixed = "#7570b3"   # Purple
        c_neural = "#2b5c8f"  # Blue

        ax.barh(y_pos, logic_pcts, color=c_logic, edgecolor="#222222", linewidth=0.8, label="Pure Logic (≥ 90%)")
        ax.barh(y_pos, mixed_pcts, left=logic_pcts, color=c_mixed, edgecolor="#222222", linewidth=0.8, label="Mixed Blend (10% - 90%)")
        ax.barh(y_pos, neural_pcts, left=logic_pcts + mixed_pcts, color=c_neural, edgecolor="#222222", linewidth=0.8, label="Pure Neural (≥ 90%)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Proportion of Dataset Transitions (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"BlendRL Decision-Making Authority Breakdown across Problems ({clean_exp})", fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        for idx in range(len(models)):
            l_val = logic_pcts[idx]
            n_val = neural_pcts[idx]
            if l_val > 10:
                ax.text(l_val / 2, idx, f"{l_val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            if n_val > 10:
                ax.text(100 - (n_val / 2), idx, f"{n_val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, framealpha=0.95, fontsize=10)
        fig.tight_layout()

        plot_path = output_dir / "blend_routing_authority.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot 1: {plot_path}")

    def _plot_routing_by_tier(self, tier_rows: list, output_dir: Path, clean_exp: str):
        """Plot 2: Grouped Bar Chart of Logic vs Neural Authority by Tier."""
        df = pd.DataFrame(tier_rows)
        if df.empty:
            return

        models = df["Model"].unique().tolist()
        fig, axes = plt.subplots(1, 3, figsize=(16, max(5.0, len(models) * 0.4)), sharey=True)

        tiers = ["Low Tier", "Med Tier", "High Tier"]
        c_logic = "#d95f02"
        c_neural = "#2b5c8f"

        for ax_idx, t_name in enumerate(tiers):
            ax = axes[ax_idx]
            sub_df = df[df["Tier"] == t_name].set_index("Model").reindex(models).fillna(0.0)
            y_pos = np.arange(len(models))
            bar_h = 0.38

            l_means = sub_df["Mean Logic Weight"].values * 100.0
            n_means = sub_df["Mean Neural Weight"].values * 100.0

            ax.barh(y_pos - bar_h/2, l_means, height=bar_h, color=c_logic, edgecolor="#222222", linewidth=0.6, label="Logic Authority ($w_{\\mathrm{logic}}$)" if ax_idx == 0 else "")
            ax.barh(y_pos + bar_h/2, n_means, height=bar_h, color=c_neural, edgecolor="#222222", linewidth=0.6, label="Neural Authority ($w_{\\mathrm{neural}}$)" if ax_idx == 0 else "")

            ax.set_title(f"{t_name}", fontsize=12, fontweight="bold")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_xlabel("Mean Authority Weight (%)", fontsize=10, fontweight="bold")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        fig.legend(["Logic Authority ($w_{\\mathrm{logic}}$)", "Neural Authority ($w_{\\mathrm{neural}}$)"],
                   loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=11, framealpha=0.95)
        fig.suptitle(f"BlendRL Neuro-Symbolic Authority by Student Competency Tier ({clean_exp})", fontsize=13, fontweight="bold", y=1.09)
        fig.tight_layout()

        plot_path = output_dir / "blend_routing_by_tier.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot 2: {plot_path}")

    def _plot_state_space_boundaries(self, state_space_data: dict, output_dir: Path, clean_exp: str):
        """Plot 4: 2D State Space Decision Boundaries via Simplex, PCA, and t-SNE."""
        if not state_space_data:
            return

        target_name = "problem" if "problem" in state_space_data else list(state_space_data.keys())[0]
        data = state_space_data[target_name]
        sub_states = data["states"]
        sub_w = data["w_logic"]

        # 1. Pedagogical Competency Simplex (GMM Posteriors)
        gmm_path = Path(f"in/datasets/pyrenees/per_problem/{target_name}/gmm_scaler.npz")
        if not gmm_path.exists():
            gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")

        have_gmm = False
        if gmm_path.exists():
            try:
                gdata = np.load(gmm_path, allow_pickle=True)
                means = gdata["means"]
                precisions = gdata["precisions"]
                log_dets = gdata["log_dets"]
                log_weights = gdata["log_weights"]
                feat_idx = gdata["feature_indices"]

                x_feat = sub_states[:, feat_idx]
                d = x_feat.shape[-1]
                const = 0.5 * d * np.log(2.0 * np.pi)
                log_probs = []
                for k in range(3):
                    diff = x_feat - means[k]
                    maha = np.sum(diff * (diff @ precisions[k]), axis=-1)
                    log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
                    log_probs.append(log_p)
                posteriors = np.exp(np.stack(log_probs, -1) - np.max(np.stack(log_probs, -1), axis=-1, keepdims=True))
                posteriors /= posteriors.sum(axis=-1, keepdims=True)

                x_mastery = posteriors[:, 2] - posteriors[:, 0]  # P(High) - P(Low) in [-1, +1]
                y_med = posteriors[:, 1]                          # P(Med) in [0, 1]
                have_gmm = True
            except Exception:
                have_gmm = False

        # 2. PCA Projection across all dimensions
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(sub_states)

        # 3. Non-linear Manifold (t-SNE)
        tsne = TSNE(n_components=2, random_state=42, perplexity=35, n_iter=1000)
        x_tsne = tsne.fit_transform(sub_states)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        cmap = LinearSegmentedColormap.from_list("NeuralLogic", ["#2b5c8f", "#9970ab", "#d95f02"])

        # Panel 1: Competency Simplex
        if have_gmm:
            sc1 = axes[0].scatter(x_mastery, y_med, c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none")
            axes[0].set_title("A. Pedagogical Competency Continuum", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Competency Continuum [P(High) − P(Low)]\n(← Struggling  |  Mastery →)", fontsize=10, fontweight="bold")
            axes[0].set_ylabel("Intermediate Uncertainty [P(Med)]", fontsize=10, fontweight="bold")
        else:
            x_feat_raw = sub_states[:, 84] if sub_states.shape[1] > 84 else sub_states[:, 0]
            y_feat_raw = sub_states[:, 72] if sub_states.shape[1] > 72 else sub_states[:, 1]
            sc1 = axes[0].scatter(x_feat_raw, y_feat_raw, c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none")
            axes[0].set_title("A. Feature Decision Space", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Steps Since Last Error", fontsize=10, fontweight="bold")
            axes[0].set_ylabel("Cumulative Accuracy", fontsize=10, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.3)

        # Panel 2: PCA
        sc2 = axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none")
        axes[1].set_title(f"B. Global State PCA (Expl. Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.3)

        # Panel 3: t-SNE
        sc3 = axes[2].scatter(x_tsne[:, 0], x_tsne[:, 1], c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none")
        axes[2].set_title("C. Non-Linear Manifold (t-SNE)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
        axes[2].set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.3)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc1, cax=cbar_ax)
        cbar.set_label("Logic Authority Weight ($w_{\\mathrm{logic}}$)", fontsize=10, fontweight="bold")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0.0 (Pure Neural)", "0.5 (Blended)", "1.0 (Pure Logic)"])

        fig.suptitle(f"BlendRL Neuro-Symbolic State Space Gating Boundaries ({clean_exp} - {target_name})", fontsize=13, fontweight="bold", y=1.02)
        fig.subplots_adjust(right=0.90, wspace=0.25)

        plot_path = output_dir / "blend_routing_state_space.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot 4 (Multi-View 2D State Space): {plot_path}")

    def _plot_ood_manifold(self, state_space_data: dict, output_dir: Path, clean_exp: str):
        """Generates 4-Row t-SNE Manifold comparing OOD Distance, Joint Authority, and Isolated Sub-Manifolds."""
        if not state_space_data:
            return

        targets = [k for k in ["exp426e(w)", "ex152a(w)", "problem"] if k in state_space_data]
        if not targets:
            targets = list(state_space_data.keys())[:3]

        n_targets = len(targets)
        fig, axes = plt.subplots(4, n_targets, figsize=(5.8 * n_targets, 17.5))
        if n_targets == 1:
            axes = axes.reshape(4, 1)

        cmap_authority = LinearSegmentedColormap.from_list("NeuralLogic", ["#2b5c8f", "#9970ab", "#d95f02"])
        cmap_ood = plt.cm.plasma
        c_neural = "#2b5c8f"
        c_logic = "#d95f02"
        c_bg = "#d6d6d6"

        for c_idx, p_name in enumerate(targets):
            data = state_space_data[p_name]
            states = data["states"]
            w_logic = data["w_logic"]
            w_neural = 1.0 - w_logic
            ood_score = data.get("ood_score", None)

            if ood_score is None:
                nbrs = NearestNeighbors(n_neighbors=20).fit(states)
                distances, _ = nbrs.kneighbors(states)
                ood_score = distances[:, -1]

            # Normalize OOD percentile
            ood_pct = pd.Series(ood_score).rank(pct=True).values * 100.0

            tsne = TSNE(n_components=2, random_state=42, perplexity=35, n_iter=1000)
            x_tsne = tsne.fit_transform(states)

            # Row 0: OOD Distance
            sc_ood = axes[0, c_idx].scatter(x_tsne[:, 0], x_tsne[:, 1], c=ood_pct, cmap=cmap_ood, vmin=0.0, vmax=100.0, alpha=0.7, s=14, edgecolors="none")
            axes[0, c_idx].set_title(f"{p_name}\n1. Epistemic OOD Distance", fontsize=11, fontweight="bold")
            axes[0, c_idx].set_xlabel("t-SNE Dim 1", fontsize=9)
            if c_idx == 0:
                axes[0, c_idx].set_ylabel("Epistemic Distance", fontsize=10, fontweight="bold")
            axes[0, c_idx].grid(True, linestyle="--", alpha=0.25)

            # Row 1: Joint Continuous Authority
            sc_auth = axes[1, c_idx].scatter(x_tsne[:, 0], x_tsne[:, 1], c=w_logic, cmap=cmap_authority, vmin=0.0, vmax=1.0, alpha=0.7, s=14, edgecolors="none")
            axes[1, c_idx].set_title(f"{p_name}\n2. Full Policy Gradient ($w_{{\\mathrm{{logic}}}}$)", fontsize=11, fontweight="bold")
            axes[1, c_idx].set_xlabel("t-SNE Dim 1", fontsize=9)
            if c_idx == 0:
                axes[1, c_idx].set_ylabel("Continuous Authority", fontsize=10, fontweight="bold")
            axes[1, c_idx].grid(True, linestyle="--", alpha=0.25)

            # Row 2: Isolated Neural-Dominated Sub-Manifold (w_neural > 0.5)
            mask_neural = (w_neural > 0.5)
            n_neu_pct = float(mask_neural.mean() * 100.0)
            axes[2, c_idx].scatter(x_tsne[:, 0], x_tsne[:, 1], color=c_bg, alpha=0.15, s=10, edgecolors="none", label="Inactive")
            axes[2, c_idx].scatter(x_tsne[mask_neural, 0], x_tsne[mask_neural, 1], color=c_neural, alpha=0.75, s=18, edgecolors="none", label="Neural > 0.5")
            axes[2, c_idx].set_title(f"{p_name}\n3. Neural Sub-Manifold ({n_neu_pct:.1f}% of states)", fontsize=11, fontweight="bold", color=c_neural)
            axes[2, c_idx].set_xlabel("t-SNE Dim 1", fontsize=9)
            if c_idx == 0:
                axes[2, c_idx].set_ylabel("Neural Sub-Manifold\n($w_{\\mathrm{neural}} > 0.5$)", fontsize=10, fontweight="bold")
            axes[2, c_idx].grid(True, linestyle="--", alpha=0.25)

            # Row 3: Isolated Logic-Dominated Sub-Manifold (w_logic >= 0.5)
            mask_logic = (w_logic >= 0.5)
            n_log_pct = float(mask_logic.mean() * 100.0)
            axes[3, c_idx].scatter(x_tsne[:, 0], x_tsne[:, 1], color=c_bg, alpha=0.15, s=10, edgecolors="none", label="Inactive")
            axes[3, c_idx].scatter(x_tsne[mask_logic, 0], x_tsne[mask_logic, 1], color=c_logic, alpha=0.75, s=18, edgecolors="none", label="Logic ≥ 0.5")
            axes[3, c_idx].set_title(f"{p_name}\n4. Logic Sub-Manifold ({n_log_pct:.1f}% of states)", fontsize=11, fontweight="bold", color=c_logic)
            axes[3, c_idx].set_xlabel("t-SNE Dim 1", fontsize=9)
            if c_idx == 0:
                axes[3, c_idx].set_ylabel("Logic Sub-Manifold\n($w_{\\mathrm{logic}} \\geq 0.5$)", fontsize=10, fontweight="bold")
            axes[3, c_idx].grid(True, linestyle="--", alpha=0.25)

        # Colorbars
        cbar_ax1 = fig.add_axes([0.92, 0.76, 0.015, 0.18])
        cb1 = fig.colorbar(sc_ood, cax=cbar_ax1)
        cb1.set_label("OOD Distance Percentile (%)\n(Dark: Familiar  |  Bright: OOD)", fontsize=9, fontweight="bold")

        cbar_ax2 = fig.add_axes([0.92, 0.52, 0.015, 0.18])
        cb2 = fig.colorbar(sc_auth, cax=cbar_ax2)
        cb2.set_label("Logic Weight ($w_{\\mathrm{logic}}$)\n(Blue: Neural  |  Orange: Logic)", fontsize=9, fontweight="bold")
        cb2.set_ticks([0.0, 0.5, 1.0])
        cb2.set_ticklabels(["0.0 (Neural)", "0.5", "1.0 (Logic)"])

        fig.suptitle(f"t-SNE Manifold Decomposition: Epistemic Familiarity, Continuous Gradient & Sub-Manifolds ({clean_exp})", fontsize=13, fontweight="bold", y=0.995)
        fig.subplots_adjust(right=0.90, hspace=0.35, wspace=0.22)

        out_path = output_dir / "blend_routing_ood_manifold.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved 4-Row OOD Manifold: {out_path}")

    def _plot_ood_handover_curve(self, ood_rows: list, output_dir: Path, clean_exp: str):
        """Plots the Balance of Expertise Handover Curve as states drift OOD."""
        if not ood_rows:
            return

        df = pd.DataFrame(ood_rows)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

        q_labels = sorted(df["OOD_Quantile"].unique())
        x_ticks = np.arange(len(q_labels))
        x_label_names = ["Q1\nIn-Distribution\n(0-20%)", "Q2\nTypical\n(20-40%)", "Q3\nBoundary\n(40-60%)", "Q4\nNear-OOD\n(60-80%)", "Q5\nFar-OOD\n(80-100%)"]

        # Panel 1: Mean Weight Curves
        logic_means = df.groupby("OOD_Quantile")["Mean_Logic_Weight"].mean().reindex(q_labels).values * 100.0
        logic_sems = df.groupby("OOD_Quantile")["Mean_Logic_Weight"].sem().reindex(q_labels).values * 100.0

        neural_means = df.groupby("OOD_Quantile")["Mean_Neural_Weight"].mean().reindex(q_labels).values * 100.0
        neural_sems = df.groupby("OOD_Quantile")["Mean_Neural_Weight"].sem().reindex(q_labels).values * 100.0

        # Plot overall mean curves
        axes[0].plot(x_ticks, logic_means, color="#d95f02", marker="o", markersize=7, linewidth=3.0, label="Logic Policy ($w_{\\mathrm{logic}}$)", zorder=5)
        axes[0].fill_between(x_ticks, logic_means - logic_sems, logic_means + logic_sems, color="#d95f02", alpha=0.20, zorder=4)

        axes[0].plot(x_ticks, neural_means, color="#2b5c8f", marker="s", markersize=7, linewidth=3.0, label="Neural Policy ($w_{\\mathrm{neural}}$)", zorder=5)
        axes[0].fill_between(x_ticks, neural_means - neural_sems, neural_means + neural_sems, color="#2b5c8f", alpha=0.20, zorder=4)

        # Plot individual problem trajectories faintly
        for m_name in df["Model"].unique():
            m_df = df[df["Model"] == m_name].set_index("OOD_Quantile").reindex(q_labels)
            axes[0].plot(x_ticks, m_df["Mean_Logic_Weight"].values * 100.0, color="#d95f02", alpha=0.20, linestyle="--", linewidth=1.0)
            axes[0].plot(x_ticks, m_df["Mean_Neural_Weight"].values * 100.0, color="#2b5c8f", alpha=0.20, linestyle="--", linewidth=1.0)

        # Reference line
        axes[0].axhline(50.0, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7, label="50% Majority Threshold")

        axes[0].set_title("A. Mean Authority: Neural Optimization ➔ Symbolic Safety Fallback", fontsize=11, fontweight="bold", pad=10)
        axes[0].set_xticks(x_ticks)
        axes[0].set_xticklabels(x_label_names, fontsize=8.5, fontweight="bold")
        axes[0].set_ylabel("Mean Decision Authority (%)", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Epistemic Uncertainty Quantiles (k-NN Distance from Training Support)\n← Familiar Training Distribution    |    Unseen / Outlier States →", fontsize=9.5, fontweight="bold")
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend(loc="best", framealpha=0.92, fontsize=9.5)

        # Panel 2: Pure Logic Fallback Rate (>= 90%)
        pure_log_means = df.groupby("OOD_Quantile")["Pure_Logic_Pct"].mean().reindex(q_labels).values
        pure_log_sems = df.groupby("OOD_Quantile")["Pure_Logic_Pct"].sem().reindex(q_labels).values

        pure_neu_means = df.groupby("OOD_Quantile")["Pure_Neural_Pct"].mean().reindex(q_labels).values
        pure_neu_sems = df.groupby("OOD_Quantile")["Pure_Neural_Pct"].sem().reindex(q_labels).values

        axes[1].plot(x_ticks, pure_log_means, color="#d95f02", marker="^", markersize=7, linewidth=2.8, label="Pure Logic Decisions (≥ 90% Authority)")
        axes[1].fill_between(x_ticks, pure_log_means - pure_log_sems, pure_log_means + pure_log_sems, color="#d95f02", alpha=0.20)

        axes[1].plot(x_ticks, pure_neu_means, color="#2b5c8f", marker="v", markersize=7, linewidth=2.8, label="Pure Neural Decisions (≥ 90% Authority)")
        axes[1].fill_between(x_ticks, pure_neu_means - pure_neu_sems, pure_neu_means + pure_neu_sems, color="#2b5c8f", alpha=0.20)

        axes[1].set_title("B. Decisive Commitment: Transitions Handed Completely to One Policy", fontsize=11, fontweight="bold", pad=10)
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels(x_label_names, fontsize=8.5, fontweight="bold")
        axes[1].set_ylabel("Proportion of State Transitions (%)", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Epistemic Uncertainty Quantiles (k-NN Distance from Training Support)\n← Familiar Training Distribution    |    Unseen / Outlier States →", fontsize=9.5, fontweight="bold")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend(loc="best", framealpha=0.92, fontsize=9.5)

        fig.suptitle(f"BlendRL Neuro-Symbolic Safety Fallback: Policy Handover across Out-of-Distribution Regimes ({clean_exp})", fontsize=12.5, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = output_dir / "blend_routing_handover_curve.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Handover Curve:   {out_path}")
