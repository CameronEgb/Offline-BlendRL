#!/usr/bin/env python3
"""
plot/blend_handover.py — Balance of Expertise Handover Curve Plotter.

Visualizes how BlendRL transitions from neural optimization to symbolic safety fallback
as student states drift out-of-distribution (OOD).
"""

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot.base import BasePlotter
from plot.blend_common import discover_blendrl_checkpoints, extract_model_routing_data


class BlendHandoverPlotter(BasePlotter):
    def __init__(self):
        super().__init__("blend_handover")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "blend_routing_ood_handover.csv"
        if csv_path.exists():
            print(f"Loading existing handover data from: {csv_path}")
            df = pd.read_csv(csv_path)
            ood_rows = df.to_dict(orient="records")
        else:
            discovered = discover_blendrl_checkpoints(exp_id, group, clean_exp)
            if not discovered:
                print(f"Notice [blend_handover]: No modular BlendRL checkpoints found for '{clean_exp}'")
                return
            data = extract_model_routing_data(discovered)
            ood_rows = data["ood_handover_rows"]
            if ood_rows:
                pd.DataFrame(ood_rows).to_csv(csv_path, index=False)
                print(f"Saved OOD Handover CSV: {csv_path}")

        if not ood_rows:
            print("Notice [blend_handover]: No OOD handover data to plot.")
            return

        self._plot_handover_curve(ood_rows, output_dir, clean_exp)

    def _plot_handover_curve(self, ood_rows: list, output_dir: Path, clean_exp: str):
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

        # 50% Threshold line
        axes[0].axhline(50.0, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7, label="50% Majority Threshold")

        axes[0].set_title("A. Mean Authority: Neural Optimization ➔ Symbolic Safety Fallback", fontsize=11, fontweight="bold", pad=10)
        axes[0].set_xticks(x_ticks)
        axes[0].set_xticklabels(x_label_names, fontsize=8.5, fontweight="bold")
        axes[0].set_ylabel("Mean Decision Authority (%)", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Epistemic Uncertainty Quantiles (k-NN Distance from Training Support)\n← Familiar Training Distribution    |    Unseen / Outlier States →", fontsize=9.5, fontweight="bold")
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend(loc="best", framealpha=0.92, fontsize=9.5)

        # Panel 2: Pure Decisions (>= 90%)
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
        print(f"  Saved Handover Curve: {out_path}")
