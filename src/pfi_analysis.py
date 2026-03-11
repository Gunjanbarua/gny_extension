"""
src/pfi_analysis.py
-------------------
Pipeline Stage 4 – Permutation Feature Importance stratified by
planting density (Trees-per-Hectare, TPH).

This stage answers the question:
  "Do the features that drive volume predictions change depending on
   how densely the stand was planted?"

Workflow
--------
1. Reproduce the *identical* train/test split used in Stage 3 so that
   PFI is evaluated on exactly the same held-out test observations.
2. Load the saved RF model and compute PFI for each TPH group.
3. Load the saved SVR pipeline and repeat the same analysis.
4. Save both result tables to results/.

Source notebook: 13.pfi_by_tph.ipynb
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predictor_cols(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in config.COLS_TO_DROP and c != "age"]
    cols.append("age")
    return cols


def _reproduce_test_set(df: pd.DataFrame):
    """
    Recreate the same plot-level stratified split used in Stage 3
    (random_state=200, test_size=0.20) so PFI is on the identical
    held-out observations.
    """
    plot_info = df[["plot", "study_x"]].drop_duplicates().reset_index(drop=True)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    _, test_idx = next(sss.split(plot_info["plot"], plot_info["study_x"]))
    test_plots  = plot_info.loc[test_idx, "plot"]
    return df[df["plot"].isin(test_plots)].copy()


def _compute_pfi_by_tph(model, test_df: pd.DataFrame,
                         X_test: pd.DataFrame, y_test,
                         model_name: str, output_path) -> pd.DataFrame:
    """
    Compute permutation feature importance for each TPH group and save.
    """
    results = []

    for tph_val in sorted(test_df["tph"].unique()):
        print(f"  Computing {model_name} PFI for TPH {tph_val} …")
        idx   = test_df[test_df["tph"] == tph_val].index
        X_sub = X_test.loc[idx]
        y_sub = y_test.loc[idx]

        if len(X_sub) < 10:
            print(f"    Skipping TPH {tph_val} – too few samples.")
            continue

        base_mse = mean_squared_error(y_sub, model.predict(X_sub))
        if base_mse == 0:
            print(f"    Skipping TPH {tph_val} – baseline MSE is zero.")
            continue

        pfi = permutation_importance(
            model, X_sub, y_sub,
            n_repeats=config.PFI_N_REPEATS_TPH,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )

        for i, feat in enumerate(X_sub.columns):
            results.append({
                "tph_group":            tph_val,
                "feature":              feat,
                "pct_increase_in_mse":  (pfi.importances_mean[i] / base_mse) * 100,
            })

    df_pfi = pd.DataFrame(results)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df_pfi.to_csv(output_path, index=False)
    print(f"  {model_name} TPH-PFI saved → {output_path}")

    # Print top-10 per group
    for tph_val in sorted(df_pfi["tph_group"].unique()):
        top = (
            df_pfi[df_pfi["tph_group"] == tph_val]
            .sort_values("pct_increase_in_mse", ascending=False)
            .head(10)
        )
        print(f"\n  Top 10 {model_name} features for TPH {tph_val}:")
        print(top.to_string(index=False))

    return df_pfi


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run():
    """Execute PFI-by-TPH analysis for both RF and SVR models."""
    print("\n" + "=" * 60)
    print("STAGE 4 – PFI by Planting Density (TPH)")
    print("=" * 60)

    # Load data and reproduce test split
    df = pd.read_csv(config.INPUT_DATA, low_memory=False)
    test_df        = _reproduce_test_set(df)
    predictor_cols = _predictor_cols(df)

    X_test = test_df[predictor_cols]
    y_test = test_df[config.TARGET_COL]
    print(f"Test set reproduced: {len(test_df):,} rows, "
          f"TPH values: {sorted(test_df['tph'].unique())}")

    # --- Random Forest ---
    if not config.RF_MODEL_PATH.exists():
        print(f"RF model not found at {config.RF_MODEL_PATH}. Run Stage 3 first.")
    else:
        print("\nLoading RF model …")
        rf_model = joblib.load(config.RF_MODEL_PATH)
        _compute_pfi_by_tph(
            rf_model, test_df, X_test, y_test,
            model_name="RF",
            output_path=config.PFI_RF_TPH,
        )

    # --- Support Vector Regression ---
    if not config.SVR_MODEL_PATH.exists():
        print(
            f"\nSVR model not found at {config.SVR_MODEL_PATH}. "
            "Skipping SVR PFI. Train and save an SVR pipeline to enable this step."
        )
    else:
        print("\nLoading SVR pipeline …")
        svr_pipeline = joblib.load(config.SVR_MODEL_PATH)
        _compute_pfi_by_tph(
            svr_pipeline, test_df, X_test, y_test,
            model_name="SVR",
            output_path=config.PFI_SVR_TPH,
        )

    print("\nStage 4 complete.\n")


if __name__ == "__main__":
    run()
