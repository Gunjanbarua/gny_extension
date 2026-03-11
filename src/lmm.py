"""
src/lmm.py
----------
Pipeline Stage 2 – Linear Mixed-Effects Model (LMM).

Workflow
--------
1. Load the merged modelling dataset.
2. Perform a group-based (tree-level) 80/20 train/test split.
3. Use Lasso regression (with StandardScaler) to select relevant fixed
   effects from among all LiDAR / competition-index candidates.
4. Compute Variance Inflation Factors (VIF) to confirm low collinearity.
5. Fit a LMM with random intercepts and random slopes on ``age``,
   grouped by ``unique_id`` (each tree as its own longitudinal group).
6. Evaluate the model on the held-out test set (R², MAE, RMSE, nRMSE,
   mean bias, uncertainty).
7. Run diagnostic plots (residuals vs fitted, Q-Q).
8. Save predictions and the fitted model summary.

Source notebook: 9.linear_mixed_effect2.ipynb
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_metrics(label: str, y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = (rmse / y_true.mean()) * 100
    bias  = np.mean(y_pred - y_true)
    unc   = np.std(y_pred  - y_true)

    print(f"\n--- {label} ---")
    print(f"  R²:          {r2_score(y_true, y_pred):.4f}")
    print(f"  MAE:         {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  nRMSE:       {nrmse:.2f}%")
    print(f"  Mean Bias:   {bias:.4f}")
    print(f"  Uncertainty: {unc:.4f}")


# ---------------------------------------------------------------------------
# Stage 2a – Train / test split (tree-group based)
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, group_col=config.GROUP_COL):
    """
    Split by unique tree IDs so that all measurements of a single tree
    are entirely in either the train or test set (no data leakage).
    """
    unique_ids = df[group_col].unique()
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train = df[df[group_col].isin(train_ids)].copy()
    test  = df[df[group_col].isin(test_ids)].copy()

    print(
        f"Train: {train.shape[0]:,} rows, {len(train_ids):,} trees  |  "
        f"Test:  {test.shape[0]:,} rows, {len(test_ids):,} trees"
    )
    return train, test


# ---------------------------------------------------------------------------
# Stage 2b – Lasso variable selection
# ---------------------------------------------------------------------------

def select_features_lasso(
    train: pd.DataFrame,
    candidates=config.FEATURE_CANDIDATES,
    target=config.TARGET_COL,
):
    """
    Fit a Lasso regression (on standardised features) and return the
    subset of candidates whose coefficients are non-zero.
    """
    X = train[candidates]
    y = train[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(
        alpha=config.LASSO_ALPHA,
        random_state=config.LASSO_RANDOM,
        max_iter=config.LASSO_MAX_ITER,
    )
    lasso.fit(X_scaled, y)

    selected = np.array(candidates)[lasso.coef_ != 0].tolist()
    print(f"Lasso selected {len(selected)} features: {selected}")
    return selected


# ---------------------------------------------------------------------------
# Stage 2c – VIF check
# ---------------------------------------------------------------------------

def compute_vif(train: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors for each selected feature.
    Values > 10 suggest harmful multicollinearity.
    """
    X_vif = train[features].copy()
    X_vif["intercept"] = 1

    vif_df = pd.DataFrame({
        "feature": features,
        "VIF": [
            variance_inflation_factor(X_vif.values, i)
            for i in range(len(features))
        ],
    })
    print("\nVIF for selected features:")
    print(vif_df.to_string(index=False))
    return vif_df


# ---------------------------------------------------------------------------
# Stage 2d – Fit LMM
# ---------------------------------------------------------------------------

def fit_lmm(train: pd.DataFrame, selected_features: list):
    """
    Fit a Linear Mixed-Effects Model:

    volume_m ~ age + <selected_features>   (fixed effects)
    (1 + age | unique_id)                  (random intercept + slope)

    Uses the L-BFGS optimizer for numerical stability.
    """
    fixed_part = " + ".join(selected_features)
    formula = f"{config.TARGET_COL} ~ {config.RANDOM_SLOPE} + {fixed_part}"
    print(f"\nFitting LMM with formula:\n  {formula}")

    lmm = smf.mixedlm(
        formula=formula,
        data=train,
        groups=train[config.GROUP_COL],
        re_formula=f"~{config.RANDOM_SLOPE}",
    )
    result = lmm.fit(method=["lbfgs"])
    print(result.summary())
    return result


# ---------------------------------------------------------------------------
# Stage 2e – Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(model_fit, train: pd.DataFrame, selected_features: list):
    """
    Breusch-Pagan heteroscedasticity test + residual diagnostic plots
    (residuals vs fitted, Q-Q plot). Saves figure to figures/.
    """
    residuals   = model_fit.resid
    fitted_vals = model_fit.fittedvalues

    # Breusch-Pagan test
    exog = sm.add_constant(train[[config.RANDOM_SLOPE] + selected_features])
    lm_stat, p_val, _, _ = het_breuschpagan(residuals, exog)
    print(f"\nBreusch-Pagan test  LM={lm_stat:.4f}  p={p_val:.4f}")
    if p_val < 0.05:
        print("  → Evidence of heteroscedasticity (p < 0.05).")
    else:
        print("  → Homoscedasticity assumption holds (p ≥ 0.05).")

    # Diagnostic plots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=fitted_vals, y=residuals, ax=axes[0], alpha=0.5)
    axes[0].axhline(0, color="#404040", linestyle="--")
    axes[0].set_title("Residuals vs Fitted Values", fontsize=14)
    axes[0].set_xlabel("Fitted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].text(0.5, -0.2, "(a)", transform=axes[0].transAxes,
                 ha="center", va="center", fontsize=14)

    sm.qqplot(residuals, line="s", ax=axes[1])
    axes[1].get_lines()[0].set_color("#404040")
    axes[1].set_title("Q-Q Plot of Residuals", fontsize=14)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=12)
    axes[1].set_ylabel("Sample Quantiles", fontsize=12)
    axes[1].text(0.5, -0.2, "(b)", transform=axes[1].transAxes,
                 ha="center", va="center", fontsize=14)

    plt.tight_layout()
    out = config.FIGURES_DIR / "lme_diagnostics.jpg"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Diagnostics figure saved → {out}")


# ---------------------------------------------------------------------------
# Stage 2f – Evaluate and save
# ---------------------------------------------------------------------------

def evaluate_and_save(model_fit, test: pd.DataFrame):
    """Predict on the test set, print metrics, and save results CSV."""
    predictions = model_fit.predict(test)
    actuals     = test[config.TARGET_COL]

    _print_metrics("LMM – Test Set Metrics", actuals, predictions)

    output_cols = ["study_x", "unique_id", "age", "tph", "study_code"]
    results = test[[c for c in output_cols if c in test.columns]].copy()
    results["predicted_volume_m"] = predictions.values

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.LMM_PREDICTIONS, index=False)
    print(f"\nLMM predictions saved → {config.LMM_PREDICTIONS}")
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run():
    """Execute the full LMM pipeline stage."""
    print("\n" + "=" * 60)
    print("STAGE 2 – Linear Mixed-Effects Model")
    print("=" * 60)

    df = pd.read_csv(config.INPUT_DATA, low_memory=False)
    print(f"Loaded input data: {df.shape}")

    train, test     = split_data(df)
    selected        = select_features_lasso(train)
    compute_vif(train, selected)
    model_fit       = fit_lmm(train, selected)
    run_diagnostics(model_fit, train, selected)
    evaluate_and_save(model_fit, test)

    print("\nStage 2 complete.\n")


if __name__ == "__main__":
    run()
