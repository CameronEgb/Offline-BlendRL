#!/usr/bin/env python3
"""
plot/blend_authority.py — BlendRL Authority & Tier Breakdown Plotter.

Visualizes overall policy authority breakdown (stacked bar) and tier-based authority (grouped bar).
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


class BlendAuthorityPlotter(BasePlotter):
    def __init__(self):
        super().__init__("blend_authority")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        auth_csv = output_dir / "blend_routing_authority.csv"
        tier_csv = output_dir / "blend_routing_by_tier.csv"

        if auth_csv.exists() and tier_csv.exists():
            print(f"Loading existing authority and tier data from CSVs in: {output_dir}")
            auth_rows = pd.read_csv(auth_csv).to_dict(orient="records")
            tier_rows = pd.read_csv(tier_csv).to_dict(orient="records")
        else:
            discovered = discover_blendrl_checkpoints(exp_id, group, clean_exp)
            if not discovered:
                print(f"Notice [blend_authority]: No modular BlendRL checkpoints found for '{clean_exp}'")
                return
            data = extract_model_routing_data(discovered)
            auth_rows = data["authority_rows"]
            tier_rows = data["tier_rows"]
            if auth_rows:
                pd.DataFrame(auth_rows).to_csv(auth_csv, index=False)
                pd.DataFrame(tier_rows).to_csv(tier_csv, index=False)

        if not auth_rows:
            print("Notice [blend_authority]: No authority data to plot.")
            return

        self._plot_routing_authority(auth_rows, output_dir, clean_exp)
        self._plot_routing_by_tier(tier_rows, output_dir, clean_exp)

    def _plot_routing_authority(self, auth_rows: list, output_dir: Path, clean_exp: str):
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
        print(f"  Saved Plot: {plot_path}")

    def _plot_routing_by_tier(self, tier_rows: list, output_dir: Path, clean_exp: str):
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
        print(f"  Saved Plot: {plot_path}")
