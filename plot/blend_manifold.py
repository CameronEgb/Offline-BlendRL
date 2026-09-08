#!/usr/bin/env python3
"""
plot/blend_manifold.py — 4-Row Decomposed t-SNE Manifold Plotter.

Visualizes epistemic state familiarity vs. neuro-symbolic authority and separates
isolated sub-manifolds for Neural (>0.5) and Logic (>=0.5) to eliminate dot overlap.
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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

from plot.base import BasePlotter
from plot.blend_common import discover_blendrl_checkpoints, extract_model_routing_data


class BlendManifoldPlotter(BasePlotter):
    def __init__(self):
        super().__init__("blend_manifold")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        discovered = discover_blendrl_checkpoints(exp_id, group, clean_exp)
        if not discovered:
            print(f"Notice [blend_manifold]: No modular BlendRL checkpoints found for '{clean_exp}'")
            return

        data = extract_model_routing_data(discovered, sample_size=4000)
        state_space_data = data["state_space_data"]

        if not state_space_data:
            print("Notice [blend_manifold]: No manifold data to plot.")
            return

        self._plot_ood_manifold(state_space_data, output_dir, clean_exp)

    def _plot_ood_manifold(self, state_space_data: dict, output_dir: Path, clean_exp: str):
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
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=20).fit(states)
                distances, _ = nbrs.kneighbors(states)
                ood_score = distances[:, -1]

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

            # Row 2: Neural Sub-Manifold
            mask_neural = (w_neural > 0.5)
            n_neu_pct = float(mask_neural.mean() * 100.0)
            axes[2, c_idx].scatter(x_tsne[:, 0], x_tsne[:, 1], color=c_bg, alpha=0.15, s=10, edgecolors="none", label="Inactive")
            axes[2, c_idx].scatter(x_tsne[mask_neural, 0], x_tsne[mask_neural, 1], color=c_neural, alpha=0.75, s=18, edgecolors="none", label="Neural > 0.5")
            axes[2, c_idx].set_title(f"{p_name}\n3. Neural Sub-Manifold ({n_neu_pct:.1f}% of states)", fontsize=11, fontweight="bold", color=c_neural)
            axes[2, c_idx].set_xlabel("t-SNE Dim 1", fontsize=9)
            if c_idx == 0:
                axes[2, c_idx].set_ylabel("Neural Sub-Manifold\n($w_{\\mathrm{neural}} > 0.5$)", fontsize=10, fontweight="bold")
            axes[2, c_idx].grid(True, linestyle="--", alpha=0.25)

            # Row 3: Logic Sub-Manifold
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
        print(f"  Saved Plot: {out_path}")
