"""
src/visualization.py
--------------------
Pipeline Stage 5 – Results visualisation.

Generates publication-quality figures comparing the three models
(Random Forest, SVR, Linear Mixed-Effects) across multiple dimensions:

  Figure 1  – Growth curves: mean predicted vs field volume ± IQR
  Figure 2  – Scatter plots: field vs predicted, coloured by age, by TPH
  Figure 3  – Residual plots: residuals vs field volume, by TPH
  Figure 4  – Overlapping histograms: actual vs predicted volume distributions
  Figure 5  – nRMSE across ages (RF vs SVR line plot)
  Figure 6  – nRMSE by TPH and age (faceted bar chart)
  Figure 7  – Overall PFI bar charts (RF and SVR)
  Figure 8  – PFI by age group (line plots)
  Figure 9  – PFI by TPH group (line plots, RF and SVR side-by-side)

All figures are written to figures/ as 300-dpi JPEGs.

Source notebook: 5.data_analysis.ipynb
"""

import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
_VIRIDIS = cm.get_cmap("viridis")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, filename: str):
    out = config.FIGURES_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _load(path, **kwargs) -> pd.DataFrame | None:
    """Load a CSV, returning None with a warning if the file is missing."""
    if not Path(path).exists():
        print(f"  [skip] File not found: {path}")
        return None
    return pd.read_csv(path, **kwargs)


def _nrmse(y_true, y_pred) -> float:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return (rmse / y_true.mean()) * 100


# ---------------------------------------------------------------------------
# Figure 1 – Growth curves
# ---------------------------------------------------------------------------

def _growth_curve_panel(ax, df: pd.DataFrame, title: str, caption: str):
    """Plot mean ± IQR growth curves for field vs predicted volume."""
    df_sorted = df.sort_values(["unique_id", "age"])
    is_new    = df_sorted.groupby("unique_id")["volume_m"].diff() != 0
    is_first  = ~df_sorted["unique_id"].duplicated(keep="first")
    field_df  = df_sorted[is_new | is_first]

    pred_stats = df.groupby("age")["predicted_volume_m"].agg(
        mean_pred="mean",
        q25_pred=lambda x: x.quantile(0.25),
        q75_pred=lambda x: x.quantile(0.75),
    )
    field_stats = field_df.groupby("age")["volume_m"].agg(
        mean_actual="mean",
        q25_actual=lambda x: x.quantile(0.25),
        q75_actual=lambda x: x.quantile(0.75),
    )
    stats = pd.concat([field_stats, pred_stats], axis=1).reset_index()

    ax.plot(stats["age"], stats["mean_actual"], label="Mean Field Volume",
            color=_VIRIDIS(0.2), lw=2.5, marker="o", ms=5, ls="--")
    ax.fill_between(stats["age"], stats["q25_actual"], stats["q75_actual"],
                    color=_VIRIDIS(0.2), alpha=0.2, label="IQR (Field)")

    ax.plot(stats["age"], stats["mean_pred"], label="Mean Predicted Volume",
            color=_VIRIDIS(0.7), lw=2.5)
    ax.fill_between(stats["age"], stats["q25_pred"], stats["q75_pred"],
                    color=_VIRIDIS(0.7), alpha=0.2, label="IQR (Predicted)")

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Age (years)", fontsize=14)
    ax.set_ylabel("Volume (m³ tree⁻¹)", fontsize=14)
    ax.legend(fontsize=12)
    ax.text(0.5, -0.12, caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=14)


def plot_growth_curves():
    """Figure 1 – Growth curves for RF and SVR side-by-side."""
    print("  Figure 1: growth curves …")
    pairs = [
        (config.RF_PREDICTIONS, "Random Forest", "(a)"),
        (config.RESULTS_DIR / "svr_final.csv", "Support Vector Regression", "(b)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)

    for ax, (path, title, caption) in zip(axes, pairs):
        df = _load(path)
        if df is None:
            ax.set_title(f"{title} – data not found", fontsize=14)
            continue
        if "predicted_volume" in df.columns:
            df = df.rename(columns={"predicted_volume": "predicted_volume_m"})
        _growth_curve_panel(ax, df, title, caption)

    axes[0].set_ylabel("Volume (m³ tree⁻¹)", fontsize=14)
    fig.suptitle("Field vs Predicted Volume: Growth Curves", fontsize=20, weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, "comparison_growth_curves.jpg")


# ---------------------------------------------------------------------------
# Figure 2 – Scatter plots by TPH (RF + SVR)
# ---------------------------------------------------------------------------

def plot_scatter_by_tph():
    """Figure 2 – Field vs predicted scatter coloured by age, per TPH group."""
    print("  Figure 2: scatter by TPH …")
    df1 = _load(config.RF_PREDICTIONS)
    df2 = _load(config.RESULTS_DIR / "svr_final.csv")
    if df1 is None or df2 is None:
        return

    tph_values = sorted(df1["tph"].unique())[:3]
    lims = [-0.09, 1.4]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12),
                             gridspec_kw={"height_ratios": [1, 1.2]})
    plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.05)

    scatter_handles = []
    for row_i, (df, row_label) in enumerate([(df1, "Random Forest"),
                                             (df2, "Support Vector Regression")]):
        if "predicted_volume" in df.columns:
            df = df.rename(columns={"predicted_volume": "predicted_volume_m"})
        for col_i, tph_val in enumerate(tph_values):
            ax     = axes[row_i, col_i]
            subset = df[df["tph"] == tph_val]
            sc = ax.scatter(
                subset["volume_m"], subset["predicted_volume_m"],
                c=subset["age"], cmap="viridis", alpha=0.7, edgecolors="k"
            )
            scatter_handles.append(sc)
            ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
            ax.set_xlim(lims); ax.set_ylim(lims)
            r2 = r2_score(subset["volume_m"], subset["predicted_volume_m"])
            ax.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes,
                    fontsize=16, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
            ax.set_title(f"TPH: {tph_val}", fontsize=16)
            ax.set_xlabel("Field Volume (m³ tree⁻¹)", fontsize=14)
            ax.set_ylabel("Predicted Volume (m³ tree⁻¹)", fontsize=14)
            cap_letter = chr(97 + row_i * 3 + col_i)
            pad = -0.18 if row_i == 0 else -0.15
            ax.text(0.5, pad, f"({cap_letter})", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", ha="center", va="top")

    fig.suptitle("Random Forest", fontsize=20, y=0.92)
    fig.text(0.5, 0.465, "Support Vector Regression", ha="center", fontsize=20)
    cbar = fig.colorbar(scatter_handles[-1], ax=axes[1, :],
                        orientation="horizontal", fraction=0.04,
                        pad=0.2, shrink=0.9, aspect=40)
    cbar.set_label("Age (years)", fontsize=14)
    plt.tight_layout(rect=[0, 1, 1, 0.95])
    _save(fig, "rf_svr_scatter_tph.jpg")


# ---------------------------------------------------------------------------
# Figure 3 – Residual plots by TPH
# ---------------------------------------------------------------------------

def plot_residuals_by_tph():
    """Figure 3 – Residuals vs field volume by TPH, RF and SVR."""
    print("  Figure 3: residuals by TPH …")
    df1 = _load(config.RF_PREDICTIONS)
    df2 = _load(config.RESULTS_DIR / "svr_final.csv")
    if df1 is None or df2 is None:
        return

    for df in [df1, df2]:
        if "predicted_volume" in df.columns:
            df.rename(columns={"predicted_volume": "predicted_volume_m"}, inplace=True)
        df["residuals"] = df["volume_m"] - df["predicted_volume_m"]

    tph_values = sorted(df1["tph"].unique())[:3]
    x_lims = [-0.09, 1.4]
    y_lims = [-0.6, 0.6]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12),
                             gridspec_kw={"height_ratios": [1, 1.2]})
    plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.05)
    scatter_handles = []

    for row_i, df in enumerate([df1, df2]):
        for col_i, tph_val in enumerate(tph_values):
            ax     = axes[row_i, col_i]
            subset = df[df["tph"] == tph_val]
            sc = ax.scatter(
                subset["volume_m"], subset["residuals"],
                c=subset["age"], cmap="viridis", alpha=0.7, edgecolors="k"
            )
            scatter_handles.append(sc)
            ax.axhline(0, color="k", ls="--", alpha=0.75, zorder=0)
            ax.set_xlim(x_lims); ax.set_ylim(y_lims)
            ax.set_title(f"TPH: {tph_val}", fontsize=16)
            ax.set_xlabel("Field Volume (m³ tree⁻¹)", fontsize=14)
            ax.set_ylabel("Residuals (m³ tree⁻¹)", fontsize=14)
            cap_letter = chr(97 + row_i * 3 + col_i)
            pad = -0.18 if row_i == 0 else -0.15
            ax.text(0.5, pad, f"({cap_letter})", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", ha="center", va="top")

    fig.suptitle("Random Forest", fontsize=20, y=0.92)
    fig.text(0.5, 0.465, "Support Vector Regression", ha="center", fontsize=20)
    cbar = fig.colorbar(scatter_handles[-1], ax=axes[1, :],
                        orientation="horizontal", fraction=0.04,
                        pad=0.2, shrink=0.9, aspect=40)
    cbar.set_label("Age (years)", fontsize=14)
    plt.tight_layout(rect=[0, 1, 1, 0.95])
    _save(fig, "rf_svr_residuals_tph.jpg")


# ---------------------------------------------------------------------------
# Figure 4 – Overlapping histograms
# ---------------------------------------------------------------------------

def plot_overlapping_histograms():
    """Figure 4 – Actual vs predicted volume distributions (RF and SVR)."""
    print("  Figure 4: overlapping histograms …")
    df_rf  = _load(config.RF_PREDICTIONS)
    df_svr = _load(config.RESULTS_DIR / "svr_final.csv")
    if df_rf is None or df_svr is None:
        return

    c_actual = _VIRIDIS(0.25)
    c_pred   = _VIRIDIS(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, df, title, caption in zip(
        axes,
        [df_rf, df_svr],
        ["Random Forest", "Support Vector Regression"],
        ["(a)", "(b)"],
    ):
        pred_col = "predicted_volume_m" if "predicted_volume_m" in df.columns else "predicted_volume"
        sns.histplot(df["volume_m"], color=c_actual, label="Actual", kde=True,
                     alpha=0.7, bins=30, ax=ax)
        sns.histplot(df[pred_col], color=c_pred, label="Predicted", kde=True,
                     alpha=0.5, bins=30, ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Volume (m³ tree⁻¹)", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.text(0.5, -0.15, caption, transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.legend()

    plt.tight_layout()
    _save(fig, "overlapping_hist.jpg")


# ---------------------------------------------------------------------------
# Figure 5 – nRMSE by age (RF vs SVR line)
# ---------------------------------------------------------------------------

def plot_nrmse_by_age():
    """Figure 5 – nRMSE across age classes for RF and SVR."""
    print("  Figure 5: nRMSE by age …")
    df_rf  = _load(config.RF_PREDICTIONS)
    df_svr = _load(config.RESULTS_DIR / "svr_final.csv")
    if df_rf is None or df_svr is None:
        return

    records = []
    for df, model in [(df_rf, "RF"), (df_svr, "SVR")]:
        pred_col = "predicted_volume_m" if "predicted_volume_m" in df.columns else "predicted_volume"
        for age_val, grp in df.groupby("age"):
            records.append({
                "age": age_val,
                "Model": model,
                "nRMSE": _nrmse(grp["volume_m"], grp[pred_col]),
            })

    df_plot = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_plot, x="age", y="nRMSE", hue="Model", marker="o", ax=ax)
    ax.set_title("nRMSE Across Ages for RF and SVR")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("nRMSE (%)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    _save(fig, "nrmse_by_age_rf_svr.jpg")


# ---------------------------------------------------------------------------
# Figure 6 – nRMSE by TPH × age (faceted bar)
# ---------------------------------------------------------------------------

def plot_nrmse_by_tph_age():
    """Figure 6 – nRMSE broken down by TPH and age for RF vs SVR."""
    print("  Figure 6: nRMSE by TPH × age …")
    df_rf  = _load(config.RF_PREDICTIONS)
    df_svr = _load(config.RESULTS_DIR / "svr_final.csv")
    if df_rf is None or df_svr is None:
        return

    df_rf["model"]  = "Random Forest"
    df_svr["model"] = "Support Vector Regression"
    if "predicted_volume" in df_svr.columns:
        df_svr = df_svr.rename(columns={"predicted_volume": "predicted_volume_m"})
    combined = pd.concat([df_rf, df_svr], ignore_index=True)

    def _nrmse_group(g):
        return _nrmse(g["volume_m"], g["predicted_volume_m"])

    perf = (
        combined.groupby(["tph", "age", "model"])
        .apply(_nrmse_group, include_groups=False)
        .reset_index(name="nRMSE")
    )

    g = sns.catplot(data=perf, x="age", y="nRMSE", hue="tph",
                    col="model", kind="bar", palette="viridis",
                    height=5, aspect=0.8)
    for i, ax in enumerate(g.axes.flat):
        ax.text(0.5, -0.15, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, ha="center", va="top")
    g.fig.subplots_adjust(bottom=0.12)
    g.set_axis_labels("Age", "nRMSE (%)")
    g.set_titles("{col_name}")
    g.legend.set_title("TPH")
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.45, 0.85),
                    ncol=3, title="TPH")
    _save(g.fig, "model_nrmse_tph_age.jpg")


# ---------------------------------------------------------------------------
# Figure 7 – Overall PFI bar charts
# ---------------------------------------------------------------------------

_TOP_FEATURES_RF  = ["vol5", "CI_Z", "Z", "Carea", "sfa5",
                      "CI_Carea", "CI_mCDst", "SILVA1", "sfa2", "mCDst"]
_TOP_FEATURES_SVR = ["CI_Z", "sfa2", "vol5", "Carea", "sfa5",
                      "vol2", "CI_mCDst", "Z", "CI_Carea", "sfa1"]


def _pfi_bar(ax, df_pfi: pd.DataFrame, features: list, title: str, caption: str):
    subset = (df_pfi[df_pfi["feature"].isin(features)]
              .sort_values("pct_increase_in_mse", ascending=False))
    sns.barplot(data=subset, x="pct_increase_in_mse", y="feature",
                palette="viridis", ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("% Increase in MSE", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    ax.grid(axis="x", ls="--", lw=0.6)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=12)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.text(0.5, -0.1, caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=14)


def plot_overall_pfi():
    """Figure 7 – Overall PFI for RF (and SVR if available)."""
    print("  Figure 7: overall PFI …")
    df_rf  = _load(config.PFI_RF_OVERALL)
    df_svr = _load(config.RESULTS_DIR / "pfi_svr_overall_final.csv")

    panels = []
    if df_rf  is not None: panels.append((df_rf,  _TOP_FEATURES_RF,  "RF",  "(a)"))
    if df_svr is not None: panels.append((df_svr, _TOP_FEATURES_SVR, "SVR", "(b)"))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(14 * len(panels), 10))
    if len(panels) == 1:
        axes = [axes]

    for ax, (df_pfi, feats, model, cap) in zip(axes, panels):
        _pfi_bar(ax, df_pfi, feats, f"Overall PFI – {model}", cap)

    plt.tight_layout()
    _save(fig, "pfi_overall_rf_svr.jpg")


# ---------------------------------------------------------------------------
# Figure 8 – PFI by age group
# ---------------------------------------------------------------------------

def plot_pfi_by_age():
    """Figure 8 – PFI trend lines across age groups for RF and SVR."""
    print("  Figure 8: PFI by age …")
    pairs = [
        (config.PFI_RF_AGE,   _TOP_FEATURES_RF,  "RF"),
        (config.RESULTS_DIR / "pfi_svr_age_final_2.csv", _TOP_FEATURES_SVR, "SVR"),
    ]
    for path, features, model in pairs:
        df_pfi = _load(path)
        if df_pfi is None:
            continue
        subset = df_pfi[df_pfi["feature"].isin(features)]
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.lineplot(data=subset, x="age_group", y="pct_increase_in_mse",
                     hue="feature", style="feature", markers=True,
                     markersize=10, linestyle="--", dashes=False,
                     palette="viridis", ax=ax)
        ax.set_title(f"PFI by Age Group – {model}", fontsize=16)
        ax.set_xlabel("Age Group", fontsize=12)
        ax.set_ylabel("% Increase in MSE", fontsize=12)
        ax.grid(True, ls="--", lw=0.5)
        ax.legend(title="Feature", bbox_to_anchor=(1.05, 1),
                  loc="upper left", fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        _save(fig, f"pfi_{model.lower()}_age.jpg")


# ---------------------------------------------------------------------------
# Figure 9 – PFI by TPH group (RF + SVR side by side)
# ---------------------------------------------------------------------------

def plot_pfi_by_tph():
    """Figure 9 – PFI by planting density for RF and SVR."""
    print("  Figure 9: PFI by TPH …")
    df_rf  = _load(config.PFI_RF_TPH)
    df_svr = _load(config.PFI_SVR_TPH)
    if df_rf is None and df_svr is None:
        return

    tph_order = [618, 1236, 1853]

    def _prep(df, features):
        sub = df[df["feature"].isin(features)].copy()
        sub["tph_group"] = pd.Categorical(sub["tph_group"],
                                          categories=tph_order, ordered=True)
        sub = sub.sort_values("tph_group")
        sub["tph_group"] = sub["tph_group"].astype(str)
        return sub

    panels = []
    if df_rf  is not None: panels.append((_prep(df_rf,  _TOP_FEATURES_RF),  "RF"))
    if df_svr is not None: panels.append((_prep(df_svr, _TOP_FEATURES_SVR), "SVR"))

    fig, axes = plt.subplots(1, len(panels), figsize=(28, 8))
    if len(panels) == 1:
        axes = [axes]

    for ax, (df_pfi, model), caption in zip(axes, panels, ["(a)", "(b)"]):
        sns.lineplot(data=df_pfi, x="tph_group", y="pct_increase_in_mse",
                     hue="feature", style="feature", markers=True,
                     markersize=10, linestyle="--", dashes=False,
                     palette="viridis", ax=ax)
        ax.set_title(f"PFI by TPH – {model}", fontsize=16)
        ax.set_xlabel("TPH Group", fontsize=14)
        ax.set_ylabel("% Increase in MSE", fontsize=14)
        ax.grid(True, ls="--", lw=0.5)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(title="Feature", bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=14)
        ax.text(0.5, -0.12, caption, transform=ax.transAxes,
                fontsize=14, ha="center", va="top")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    _save(fig, "pfi_tph_rf_svr.jpg")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run():
    """Generate all figures."""
    print("\n" + "=" * 60)
    print("STAGE 5 – Visualisation")
    print("=" * 60)

    config.FIGURES_DIR.mkdir(exist_ok=True)

    plot_growth_curves()
    plot_scatter_by_tph()
    plot_residuals_by_tph()
    plot_overlapping_histograms()
    plot_nrmse_by_age()
    plot_nrmse_by_tph_age()
    plot_overall_pfi()
    plot_pfi_by_age()
    plot_pfi_by_tph()

    print("\nStage 5 complete.\n")


if __name__ == "__main__":
    run()
