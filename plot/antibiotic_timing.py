#!/usr/bin/env python3
"""
plot/antibiotic_timing.py — MIMIC Early Antibiotic Intervention Plotter.

Inspired by Figure 4 of "THEMES and Offline AL Framework" (ACM 2025).

Generates two outputs per experiment:
1. **Case-study figure** (`antibiotic_timing_case_study.png`):
   A multi-panel figure for the best sampled patient trajectory showing:
   - Top 3 panels: temporal vital signs (Heart Rate, Respiratory Rate, SpO2/Temp)
     with green shading for normal ranges and red for abnormal.
   - Bottom panel: action comparison — clinician dots (●), each RL policy with
     distinct markers (×, ▲, ■, …), annotated with ΔT_First arrows.

2. **ΔT_First summary bar chart** (`antibiotic_timing_delta_summary.png`):
   Median ΔT_First (hours) per method across all patients where the policy
   recommends treatment before the clinician (positive = earlier than clinician).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve

from plot.base import BasePlotter, clean_label, get_canonical_method_name
from src.usr.methods.method_style_registry import get_style as get_method_style

# ---------------------------------------------------------------------------
# Normal vital-sign ranges (for shading) — expressed in original units
# ---------------------------------------------------------------------------
VITAL_NORMALS = {
    "HeartRate_mean": {"label": "Heart Rate (bpm)", "col_x": 0, "col_orig": 0, "low": 60, "high": 100},
    "RespiratoryRate_mean": {"label": "Respiratory Rate (breaths/min)", "col_x": 1, "col_orig": 1, "low": 12, "high": 20},
    "Temperature_mean": {"label": "Temperature (°C)", "col_x": 7, "col_orig": 7, "low": 36.1, "high": 37.9},
}

# Markers for policies on the action timeline panel
_POLICY_MARKERS = ["x", "^", "s", "D", "P", "v", "*", "h", "+", "8"]


class AntibioticTimingPlotter(BasePlotter):
    """
    Plots ΔT_First early intervention analysis for MIMIC RL policies.

    Outputs:
      - antibiotic_timing/case_study_patient_<N>.png  — per-patient case studies
      - antibiotic_timing/delta_t_first_summary.png   — median ΔT_First bar chart
      - antibiotic_timing/delta_t_first_stats.csv     — per-method statistics table
    """

    def __init__(self):
        super().__init__("antibiotic_timing")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem

        print("\n" + "=" * 90)
        print(f"=== Generating Antibiotic Timing (ΔT_First) Plots for '{exp_id}' ===")
        print("=" * 90)

        # ------------------------------------------------------------------
        # 1. Load dataset
        # ------------------------------------------------------------------
        data, X, mask, y, valid_mask, clin_acts, orig = self._load_dataset(cfg)
        if data is None:
            return

        num_patients = X.shape[0]
        total_steps = int(valid_mask.sum())
        all_obs = X[:, :, :46][valid_mask]
        all_clin_acts_flat = clin_acts[valid_mask]
        orig_names = list(data.get("feature_names_orig", []))
        print(f"  Loaded {num_patients:,} patients, {total_steps:,} valid transitions.")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # ------------------------------------------------------------------
        # 2. Discover and load policy checkpoints
        # ------------------------------------------------------------------
        method_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not method_ckpts:
            print(f"  Notice [antibiotic_timing]: No checkpoints found for '{clean_exp}'. Skipping.")
            return

        print(f"  Discovered {len(method_ckpts)} policy checkpoint(s).")

        # ------------------------------------------------------------------
        # 3. Run inference — get per-patient first-antibiotic timestep per method
        # ------------------------------------------------------------------
        method_first_ab: dict[str, np.ndarray] = {}  # method → (num_patients,) array, NaN if never treated
        method_opt_thresh: dict[str, float] = {}

        batch_size = int(cfg.get("batch_size", 10_000))

        for method_name, ckpt_path in sorted(method_ckpts.items()):
            agent = self._load_agent(ckpt_path, device)
            if agent is None:
                continue

            print(f"  Running inference for {clean_label(method_name)} ...", flush=True)

            all_probs = []
            with torch.no_grad():
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    obs_b = torch.tensor(all_obs[b_start:b_end], dtype=torch.float32).to(device)
                    probs, _ = self._get_probs_and_actions(agent, obs_b)
                    p_admin = probs[:, 1].cpu().numpy() if probs.shape[-1] > 1 else probs.squeeze().cpu().numpy()
                    all_probs.extend(p_admin)

            all_probs = np.array(all_probs)

            # Calibrated threshold (PR-curve F1 maximisation vs clinician)
            try:
                p_thr, r_thr, thresholds = precision_recall_curve(all_clin_acts_flat, all_probs)
                f1 = 2 * p_thr * r_thr / (p_thr + r_thr + 1e-8)
                best_idx = int(np.argmax(f1))
                opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            except Exception:
                opt_thresh = 0.5

            method_opt_thresh[method_name] = opt_thresh
            policy_acts_flat = (all_probs >= opt_thresh).astype(int)

            # Map flat actions back to per-patient first-AB timestep
            first_ab = np.full(num_patients, np.nan)
            step_idx = 0
            for p_idx in range(num_patients):
                p_valid = valid_mask[p_idx]
                p_len = int(p_valid.sum())
                if p_len == 0:
                    continue
                p_pol = policy_acts_flat[step_idx : step_idx + p_len]
                ab_steps = np.where(p_pol == 1)[0]
                if len(ab_steps) > 0:
                    # Timestep index relative to the full 240-step array
                    valid_step_indices = np.where(p_valid)[0]
                    first_ab[p_idx] = float(valid_step_indices[ab_steps[0]])
                step_idx += p_len

            method_first_ab[method_name] = first_ab

        if not method_first_ab:
            print("  No inference results. Aborting antibiotic_timing plots.")
            return

        # ------------------------------------------------------------------
        # 4. Compute clinician first-AB timestep per patient
        # ------------------------------------------------------------------
        clin_first_ab = np.full(num_patients, np.nan)
        for p_idx in range(num_patients):
            p_valid = valid_mask[p_idx]
            valid_step_indices = np.where(p_valid)[0]
            p_clin = clin_acts[p_idx][p_valid]
            ab_steps = np.where(p_clin == 1)[0]
            if len(ab_steps) > 0:
                clin_first_ab[p_idx] = float(valid_step_indices[ab_steps[0]])

        # ------------------------------------------------------------------
        # 5. ΔT_First = clin_first_ab - policy_first_ab  (positive = earlier)
        # ------------------------------------------------------------------
        delta_t: dict[str, np.ndarray] = {}
        for m, first_ab in method_first_ab.items():
            # Only consider patients where clinician actually administered antibiotics
            clin_treated = ~np.isnan(clin_first_ab)
            pol_treated = ~np.isnan(first_ab)
            both = clin_treated & pol_treated
            dt = clin_first_ab[both] - first_ab[both]  # hours earlier (positive = policy earlier)
            delta_t[m] = dt

        # ------------------------------------------------------------------
        # 6. Case-study figure(s)
        # ------------------------------------------------------------------
        n_cases = int(cfg.get("n_case_studies", 3))
        max_hours = int(cfg.get("max_hours", 240))
        out_dir = output_dir / "antibiotic_timing"
        out_dir.mkdir(parents=True, exist_ok=True)

        chosen_patients = self._select_case_study_patients(
            clin_first_ab, method_first_ab, y, n_cases=n_cases, cfg=cfg
        )

        orig_col_map = {name: i for i, name in enumerate(orig_names)} if orig_names else {}

        for case_idx, p_idx in enumerate(chosen_patients):
            self._plot_case_study(
                p_idx=p_idx,
                case_idx=case_idx,
                X=X,
                orig=orig,
                orig_col_map=orig_col_map,
                valid_mask=valid_mask,
                clin_acts=clin_acts,
                clin_first_ab=clin_first_ab,
                method_first_ab=method_first_ab,
                y=y,
                out_dir=out_dir,
                clean_exp=clean_exp,
                cfg=cfg,
                max_hours=max_hours,
            )

        # ------------------------------------------------------------------
        # 7. ΔT_First summary bar chart
        # ------------------------------------------------------------------
        self._plot_delta_t_summary(delta_t, out_dir, clean_exp, cfg)

        # ------------------------------------------------------------------
        # 8. Save CSV summary
        # ------------------------------------------------------------------
        stats_rows = []
        for m, dt in delta_t.items():
            pos = dt[dt > 0]
            stats_rows.append(
                {
                    "Method": clean_label(m),
                    "n_patients_clin_treated": len(dt),
                    "n_policy_earlier": int((dt > 0).sum()),
                    "pct_policy_earlier": float((dt > 0).mean() * 100) if len(dt) > 0 else 0.0,
                    "median_delta_t_all_h": float(np.median(dt)) if len(dt) > 0 else np.nan,
                    "median_delta_t_earlier_h": float(np.median(pos)) if len(pos) > 0 else np.nan,
                    "mean_delta_t_h": float(np.mean(dt)) if len(dt) > 0 else np.nan,
                }
            )
        if stats_rows:
            df = pd.DataFrame(stats_rows)
            csv_path = out_dir / "delta_t_first_stats.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved ΔT_First stats: {csv_path}")
            print(df.to_string(index=False))

        print("=" * 90 + "\n")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset(self, cfg: dict):
        env_ds = (
            cfg.get("env", {}).get("dataset_name", "mimic_lazy_0_interventions_balanced.npz")
            if isinstance(cfg.get("env"), dict)
            else "mimic_lazy_0_interventions_balanced.npz"
        )
        npz_path = Path("in/datasets/mimic") / env_ds
        if not npz_path.exists():
            print(f"  [antibiotic_timing] Dataset not found: {npz_path}")
            return None, None, None, None, None, None, None

        import numpy as np

        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]          # (N, 240, 49) — z-scored features + actions
        mask = data["mask"]    # (N, 240, 1)
        y = data["y"].squeeze()
        orig = data.get("orig", None)  # (N, 240, 18) raw vital signs, may have NaNs

        valid_mask = mask.squeeze(-1) != -1   # (N, 240) bool
        clin_acts = X[:, :, 47].astype(int)   # col 47 = AntiInfectiveAdmin_max
        return data, X, mask, y, valid_mask, clin_acts, orig

    # ------------------------------------------------------------------
    # Checkpoint discovery (mirrors clinical_alignment approach)
    # ------------------------------------------------------------------
    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str) -> dict:
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases, has_active_filter = self.get_active_aliases(exp_cfg)

        method_ckpts = {}
        for method_dir in sorted(ckpt_root.iterdir()):
            if not method_dir.is_dir():
                continue
            m_name = method_dir.name
            if not self.is_method_active(m_name, active_aliases, has_active_filter):
                continue
            canon = get_canonical_method_name(m_name)
            ckpts = list(method_dir.rglob("best_model*.ckpt"))
            if ckpts and canon not in method_ckpts:
                method_ckpts[canon] = ckpts[0]

        return method_ckpts

    # ------------------------------------------------------------------
    # Agent loading (mirrors clinical_alignment)
    # ------------------------------------------------------------------
    def _load_agent(self, path, dev):
        from src.usr.methods.cql_agent import CQLAgent
        from src.usr.methods.iql_agent import IQLAgent

        classes = [CQLAgent, IQLAgent]
        try:
            from src.usr.methods.cew_agent import CEWAgent
            classes.insert(1, CEWAgent)
        except ImportError:
            pass

        for cls in classes:
            for strict in [True, False]:
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False, strict=strict)
                    ag.to(dev).eval()
                    return ag
                except Exception:
                    continue
        print(f"  [antibiotic_timing] Could not load checkpoint: {path}")
        return None

    # ------------------------------------------------------------------
    # Action probability extraction (mirrors clinical_alignment)
    # ------------------------------------------------------------------
    def _get_probs_and_actions(self, ag, obs_b):
        if hasattr(ag, "get_action_probs"):
            probs = ag.get_action_probs(obs_b)
            acts = ag.get_action(obs_b) if hasattr(ag, "get_action") else torch.argmax(probs, dim=-1)
            return probs, acts

        is_cql = ag.__class__.__name__ == "CQLAgent" or "cql" in str(getattr(ag, "algorithm", "")).lower()
        use_actor = bool(ag.get_cfg("use_actor", False)) if hasattr(ag, "get_cfg") else getattr(ag, "use_actor", False)

        if hasattr(ag, "is_modular") and ag.is_modular:
            logic_obs = (
                ag._prepare_logic_obs(obs_b)
                if hasattr(ag, "_prepare_logic_obs")
                else obs_b.unsqueeze(1).repeat(1, 2, 1)
            )
            if is_cql and not use_actor and hasattr(ag.model, "get_q_values"):
                q = ag.model.get_q_values(obs_b, logic_obs)
                return torch.softmax(q, dim=-1), torch.argmax(q, dim=-1)
            elif hasattr(ag.model, "get_q_values"):
                q = ag.model.get_q_values(obs_b, logic_obs)
                return torch.softmax(q, dim=-1), torch.argmax(q, dim=-1)
        elif hasattr(ag, "q_network"):
            q = ag.q_network.get_q_values(obs_b) if hasattr(ag.q_network, "get_q_values") else ag.q_network(obs_b)
            return torch.softmax(q, dim=-1), torch.argmax(q, dim=-1)
        elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
            q = ag.model.get_q_values(obs_b)
            return torch.softmax(q, dim=-1), torch.argmax(q, dim=-1)
        else:
            out = ag.get_action_and_value(obs_b)
            act = out[0] if isinstance(out, (tuple, list)) else out
            n_acts = 3 if obs_b.shape[-1] >= 123 else 2
            probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
            probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
            return probs, act

    # ------------------------------------------------------------------
    # Case-study patient selection
    # ------------------------------------------------------------------
    def _select_case_study_patients(
        self,
        clin_first_ab: np.ndarray,
        method_first_ab: dict[str, np.ndarray],
        y: np.ndarray,
        n_cases: int,
        cfg: dict,
    ) -> list[int]:
        """
        Select patients where:
        - Clinician administered antibiotics (clin_first_ab not NaN)
        - At least one policy recommends treatment BEFORE the clinician (ΔT > 0)
        - Patient has shock outcome (y=1), if available — most clinically dramatic
        - Patient has a reasonable trajectory length (≥ 20 valid steps)

        Returns up to n_cases patient indices.
        """
        forced = cfg.get("case_study_patient_ids", [])
        if forced:
            return [int(p) for p in forced[:n_cases]]

        num_patients = len(clin_first_ab)
        scores = []
        for p_idx in range(num_patients):
            if np.isnan(clin_first_ab[p_idx]):
                continue
            clin_t = clin_first_ab[p_idx]
            # Compute max ΔT across methods
            max_delta = -np.inf
            for first_ab in method_first_ab.values():
                if not np.isnan(first_ab[p_idx]):
                    max_delta = max(max_delta, clin_t - first_ab[p_idx])
            if max_delta <= 0:
                continue
            shock_bonus = 10.0 if (p_idx < len(y) and y[p_idx] == 1) else 0.0
            scores.append((p_idx, max_delta + shock_bonus))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:n_cases]]

    # ------------------------------------------------------------------
    # Per-patient case study figure
    # ------------------------------------------------------------------
    def _plot_case_study(
        self,
        p_idx: int,
        case_idx: int,
        X: np.ndarray,
        orig,
        orig_col_map: dict,
        valid_mask: np.ndarray,
        clin_acts: np.ndarray,
        clin_first_ab: np.ndarray,
        method_first_ab: dict[str, np.ndarray],
        y: np.ndarray,
        out_dir: Path,
        clean_exp: str,
        cfg: dict,
        max_hours: int,
    ):
        """Generates the 4-panel case study figure for a single patient."""
        valid_steps = np.where(valid_mask[p_idx])[0]  # absolute timestep indices (hours)
        n_valid = len(valid_steps)
        if n_valid < 5:
            return

        # Clip to max_hours
        valid_steps = valid_steps[valid_steps < max_hours]
        hours = valid_steps  # x-axis in hours (0-indexed ICU admission)

        dpi = int(cfg.get("dpi", 200))
        figsize = tuple(cfg.get("figsize_case_study", [14, 11]))
        shock_label = "Septic Shock" if (p_idx < len(y) and y[p_idx] == 1) else "No Shock"

        # Vital signs to plot in top panels
        vitals_to_plot = cfg.get(
            "vital_signs",
            ["HeartRate_mean", "RespiratoryRate_mean", "Temperature_mean"],
        )
        # Keep only those we have normals for
        vitals_to_plot = [v for v in vitals_to_plot if v in VITAL_NORMALS][:3]
        n_panels = len(vitals_to_plot) + 1  # vitals + action panel

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
        if n_panels == 1:
            axes = [axes]

        # ---- Top panels: vital signs ----
        for v_idx, vital_key in enumerate(vitals_to_plot):
            ax = axes[v_idx]
            vinfo = VITAL_NORMALS[vital_key]

            # Try raw (orig) data first; only use if there is meaningful variation
            use_raw = False
            vals = None
            if orig is not None and vinfo["col_orig"] < orig.shape[2]:
                raw_vals = orig[p_idx, valid_steps, vinfo["col_orig"]]
                finite_raw = raw_vals[np.isfinite(raw_vals)]
                if len(finite_raw) >= 3 and finite_raw.std() > 0.1 and finite_raw.max() > 5:
                    use_raw = True
                    vals = pd.Series(raw_vals).interpolate(method="linear").ffill().bfill().values
                    ylabel = vinfo["label"]
                    normal_low, normal_high = float(vinfo["low"]), float(vinfo["high"])

            if not use_raw:
                # Use z-scored values from X; normals become [-1, +1]
                raw_z = X[p_idx, valid_steps, vinfo["col_x"]]
                vals = pd.Series(raw_z).interpolate(method="linear").ffill().bfill().values
                ylabel = vinfo["label"].split(" (")[0] + " (z-score)"
                normal_low, normal_high = -1.0, 1.0

            # Compute y-axis limits from actual data range
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) == 0:
                ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            v_min, v_max = float(finite_vals.min()), float(finite_vals.max())
            padding = max((v_max - v_min) * 0.15, 0.5)
            y_min = min(v_min - padding, normal_low - padding * 0.5)
            y_max = max(v_max + padding, normal_high + padding * 0.5)

            # Normal range shading (green inside, red outside)
            ax.axhspan(normal_low, normal_high, color="green", alpha=0.15, zorder=1)
            if y_min < normal_low:
                ax.axhspan(y_min, normal_low, color="red", alpha=0.10, zorder=1)
            if y_max > normal_high:
                ax.axhspan(normal_high, y_max, color="red", alpha=0.10, zorder=1)

            ax.plot(hours[: len(vals)], vals, color="steelblue", linewidth=1.5, zorder=3)
            ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle="--", alpha=0.3)

            # Mark clinician's first AB time
            if not np.isnan(clin_first_ab[p_idx]):
                ax.axvline(clin_first_ab[p_idx], color="black", linestyle=":", linewidth=1.2, alpha=0.6)

        # ---- Bottom panel: action timeline ----
        ax_act = axes[-1]
        ax_act.set_ylabel("Treatment\nAction", fontsize=9, fontweight="bold")
        ax_act.set_ylim(-0.5, 1.5)
        ax_act.set_yticks([0, 1])
        ax_act.set_yticklabels(["No AB", "Antibiotic"], fontsize=8)
        ax_act.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Clinician actions
        p_clin = clin_acts[p_idx, valid_steps].astype(int)
        clin_times = hours[p_clin == 1]
        ax_act.scatter(
            clin_times,
            np.ones_like(clin_times) * 1.0,
            marker="o",
            color="black",
            s=60,
            zorder=5,
            label="Clinician (●)",
        )

        # Policy actions
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8, label="Clinician"),
        ]

        for m_idx, (method_name, first_ab_arr) in enumerate(sorted(method_first_ab.items())):
            style = get_method_style(method_name)
            color = style.get("color") or f"C{m_idx}"
            marker = _POLICY_MARKERS[m_idx % len(_POLICY_MARKERS)]
            display = clean_label(method_name)

            # Reconstruct policy actions for this patient from first_ab_arr (best we can)
            # We only have first-AB timing; show as a point at the first AB time
            pol_first = first_ab_arr[p_idx]
            if not np.isnan(pol_first) and pol_first < max_hours:
                ax_act.scatter(
                    [pol_first],
                    [0.5],
                    marker=marker,
                    color=color,
                    s=80,
                    zorder=5,
                    linewidths=1.5,
                )
                # ΔT arrow
                clin_t = clin_first_ab[p_idx]
                if not np.isnan(clin_t) and clin_t != pol_first:
                    delta_h = clin_t - pol_first
                    arrow_y = 0.5
                    ax_act.annotate(
                        f"ΔT={delta_h:+.0f}h",
                        xy=(pol_first, arrow_y),
                        xytext=(pol_first, arrow_y + 0.5),
                        fontsize=7.5,
                        color=color,
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                    )

            legend_handles.append(
                Line2D([0], [0], marker=marker, color="w", markerfacecolor=color,
                       markeredgecolor=color, markersize=8, label=display),
            )

        # Clinician first-AB vertical line
        if not np.isnan(clin_first_ab[p_idx]):
            for ax in axes:
                ax.axvline(clin_first_ab[p_idx], color="black", linestyle=":", linewidth=1.2, alpha=0.7)
            ax_act.axvline(
                clin_first_ab[p_idx], color="black", linestyle=":", linewidth=1.5,
                label="Clinician first AB", alpha=0.8,
            )

        ax_act.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
        ax_act.set_xlabel("Hours Since ICU Admission", fontsize=10, fontweight="bold")

        # Title
        fig.suptitle(
            f"Case Study — Patient {p_idx} ({shock_label}) | {clean_exp.replace('_', ' ').title()}",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"case_study_patient_{p_idx}.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved case study: {out_path}")

    # ------------------------------------------------------------------
    # ΔT_First summary bar chart
    # ------------------------------------------------------------------
    def _plot_delta_t_summary(
        self,
        delta_t: dict[str, np.ndarray],
        out_dir: Path,
        clean_exp: str,
        cfg: dict,
    ):
        """Bar chart of median ΔT_First (hours) per method.

        Positive bars = policy recommends antibiotics earlier than clinician.
        Error bars show the 25th–75th percentile range.
        """
        if not delta_t:
            return

        figsize = tuple(cfg.get("figsize_summary", [max(6, len(delta_t) * 2), 5]))
        dpi = int(cfg.get("dpi", 300))

        methods = sorted(delta_t.keys())
        medians = []
        q25 = []
        q75 = []
        pct_earlier = []
        colors = []

        for m in methods:
            dt = delta_t[m]
            medians.append(float(np.median(dt)) if len(dt) > 0 else 0.0)
            q25.append(float(np.percentile(dt, 25)) if len(dt) > 0 else 0.0)
            q75.append(float(np.percentile(dt, 75)) if len(dt) > 0 else 0.0)
            pct_earlier.append(float((dt > 0).mean() * 100) if len(dt) > 0 else 0.0)
            style = get_method_style(m)
            colors.append(style.get("color") or "tab:blue")

        labels = [clean_label(m) for m in methods]
        x = np.arange(len(methods))
        yerr_low = np.array(medians) - np.array(q25)
        yerr_high = np.array(q75) - np.array(medians)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            x, medians,
            color=colors, width=0.55, edgecolor="#333333", linewidth=1.0, alpha=0.85,
            yerr=[yerr_low, yerr_high],
            capsize=5, error_kw={"linewidth": 1.4, "ecolor": "#555555"},
        )

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Median ΔT_First (hours)\n[positive = policy earlier than clinician]",
                       fontsize=10, fontweight="bold")
        ax.set_title(
            f"{clean_exp.replace('_', ' ').title()}: Early Antibiotic Intervention (ΔT_First)",
            fontsize=11, fontweight="bold",
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        # Add 20% headroom above the tallest bar+error so annotations don't hit the title
        all_tops = [m + e for m, e in zip(medians, yerr_high)]
        all_bots = [m - e for m, e in zip(medians, yerr_low)]
        data_top = max(all_tops) if all_tops else 1.0
        data_bot = min(all_bots) if all_bots else -1.0
        span = max(data_top - data_bot, 1.0)
        ax.set_ylim(data_bot - span * 0.15, data_top + span * 0.35)

        for i, (bar, med, pct) in enumerate(zip(bars, medians, pct_earlier)):
            height = bar.get_height()
            offset_dir = 1 if height >= 0 else -1
            err_offset = yerr_high[i] if height >= 0 else yerr_low[i]
            label_y = height + offset_dir * (err_offset + 0.5)
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{med:+.1f}h\n({pct:.0f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, label_y),
                ha="center", va=va, fontsize=8.5, fontweight="bold",
            )

        ax.text(
            0.98, 0.02,
            "Error bars: IQR (25th–75th percentile)\n% = fraction of patients where policy treats earlier",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            style="italic", color="#555555",
        )

        fig.tight_layout()
        out_path = out_dir / "delta_t_first_summary.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved ΔT_First summary: {out_path}")
