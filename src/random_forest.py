"""
src/random_forest.py
--------------------
Pipeline Stage 3 – Random Forest Regressor.

Workflow
--------
1. Load the merged modelling dataset.
2. Perform a stratified-spatial 80/20 train/test split, stratified by
   study site and split at the plot level (not tree level) to prevent
   spatial autocorrelation from inflating test-set performance.
3. Tune hyperparameters with ``HalvingGridSearchCV`` using 5-fold
   ``GroupKFold`` cross-validation (groups = plots).
4. Evaluate the best model on the held-out test set globally and
   disaggregated by TPH group and age class.
5. Compute Permutation Feature Importance (PFI) both overall and
   stratified by age group.
6. Save the trained model (joblib) and predictions (CSV).

Source notebook: 11.rf_final.ipynb
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingGridSearchCV,
    GroupKFold,
    StratifiedShuffleSplit,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label: str) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    nrmse = (rmse / y_true.mean()) * 100 if y_true.mean() != 0 else 0
    r2    = r2_score(y_true, y_pred)
    bias  = np.mean(y_pred - y_true)
    unc   = np.std(y_pred  - y_true)

    print(f"\nMetrics – {label}:")
    print(f"  R²:          {r2:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  nRMSE:       {nrmse:.2f}%")
    print(f"  Mean Bias:   {bias:.4f}")
    print(f"  Uncertainty: {unc:.4f}")

    return dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse,
                nRMSE=nrmse, Bias=bias, Uncertainty=unc)


def _predictor_cols(df: pd.DataFrame) -> list:
    """Return feature columns (all columns not in COLS_TO_DROP, plus 'age')."""
    cols = [c for c in df.columns if c not in config.COLS_TO_DROP and c != "age"]
    cols.append("age")
    return cols


# ---------------------------------------------------------------------------
# Stage 3a – Stratified spatial train / test split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """
    Split at the *plot* level, stratified by study site, so that
    spatial blocks of trees are kept together.  This mirrors a
    realistic deployment scenario where the model is applied to
    unseen locations.
    """
    plot_info = df[["plot", "study_x"]].drop_duplicates().reset_index(drop=True)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    train_idx, test_idx = next(
        sss.split(plot_info["plot"], plot_info["study_x"])
    )

    train_plots = plot_info.loc[train_idx, "plot"]
    test_plots  = plot_info.loc[test_idx,  "plot"]

    train = df[df["plot"].isin(train_plots)].copy()
    test  = df[df["plot"].isin(test_plots)].copy()

    print(
        f"Plots – train: {len(train_plots)}, test: {len(test_plots)}\n"
        f"Rows  – train: {len(train):,},    test: {len(test):,}"
    )
    print("\nTrain study distribution:\n",
          train["study_x"].value_counts(normalize=True).round(3))
    print("\nTest study distribution:\n",
          test["study_x"].value_counts(normalize=True).round(3))
    return train, test


# ---------------------------------------------------------------------------
# Stage 3b – Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random_forest(train: pd.DataFrame, predictor_cols: list):
    """
    Run Successive Halving Grid Search with GroupKFold (groups = plots)
    to find the best Random Forest hyperparameters.
    """
    X_train = train[predictor_cols]
    y_train = train[config.TARGET_COL]
    groups  = train[config.PLOT_COL]

    rf = RandomForestRegressor(
        random_state=config.RANDOM_STATE, n_jobs=-1
    )
    cv = GroupKFold(n_splits=config.RF_CV_FOLDS)

    search = HalvingGridSearchCV(
        estimator=rf,
        param_grid=config.RF_PARAM_GRID,
        cv=cv,
        factor=config.RF_CV_FACTOR,
        n_jobs=-1,
        verbose=1,
        scoring="neg_mean_squared_error",
        random_state=config.RANDOM_STATE,
    )

    print("Starting hyperparameter search …")
    search.fit(X_train, y_train, groups=groups)

    print(f"\nBest parameters: {search.best_params_}")
    best_cv_rmse = np.sqrt(-search.best_score_)
    print(f"Best CV RMSE:    {best_cv_rmse:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Stage 3c – Permutation Feature Importance
# ---------------------------------------------------------------------------

def compute_pfi_overall(model, X_test, y_test, baseline_mse: float):
    """Overall PFI on the full test set (50 repeats)."""
    pfi = permutation_importance(
        model, X_test, y_test,
        n_repeats=config.PFI_N_REPEATS_OVERALL,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    rows = [
        {
            "feature": feat,
            "pct_increase_in_mse": (pfi.importances_mean[i] / baseline_mse) * 100,
        }
        for i, feat in enumerate(X_test.columns)
    ]
    df_pfi = pd.DataFrame(rows).sort_values("pct_increase_in_mse", ascending=False)
    df_pfi.to_csv(config.PFI_RF_OVERALL, index=False)
    print(f"\nOverall PFI saved → {config.PFI_RF_OVERALL}")
    print("Top 10 features:\n", df_pfi.head(10).to_string(index=False))
    return df_pfi


def compute_pfi_by_age(model, test_df: pd.DataFrame, X_test, y_test):
    """PFI computed separately for each age class (200 repeats per class)."""
    results = []

    for age_val in sorted(test_df["age"].unique()):
        idx     = test_df[test_df["age"] == age_val].index
        X_sub   = X_test.loc[idx]
        y_sub   = y_test.loc[idx]

        if len(X_sub) < 10:
            continue

        base_mse = mean_squared_error(y_sub, model.predict(X_sub))
        if base_mse == 0:
            continue

        pfi = permutation_importance(
            model, X_sub, y_sub,
            n_repeats=config.PFI_N_REPEATS_AGE,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )
        for i, feat in enumerate(X_sub.columns):
            results.append({
                "age_group": age_val,
                "feature":   feat,
                "pct_increase_in_mse": (pfi.importances_mean[i] / base_mse) * 100,
            })
        print(f"  PFI by age – age {age_val} done.")

    df_pfi = pd.DataFrame(results)
    df_pfi.to_csv(config.PFI_RF_AGE, index=False)
    print(f"Age-stratified PFI saved → {config.PFI_RF_AGE}")
    return df_pfi


# ---------------------------------------------------------------------------
# Stage 3d – Save predictions
# ---------------------------------------------------------------------------

def save_predictions(test: pd.DataFrame, predictions: np.ndarray):
    output_cols = [
        "study_x", "plot", "tree", "unique_id",
        "tph", "age", config.TARGET_COL,
    ]
    out = test[[c for c in output_cols if c in test.columns]].copy()
    out["predicted_volume_m"] = predictions

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.RF_PREDICTIONS, index=False)
    print(f"RF predictions saved → {config.RF_PREDICTIONS}")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run():
    """Execute the full Random Forest pipeline stage."""
    print("\n" + "=" * 60)
    print("STAGE 3 – Random Forest")
    print("=" * 60)

    df            = pd.read_csv(config.INPUT_DATA, low_memory=False)
    predictor_cols = _predictor_cols(df)

    train, test   = split_data(df)
    X_train = train[predictor_cols]
    y_train = train[config.TARGET_COL]
    X_test  = test[predictor_cols]
    y_test  = test[config.TARGET_COL]

    # Tune and train
    best_model = tune_random_forest(train, predictor_cols)

    # Save model
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, config.RF_MODEL_PATH)
    print(f"Model saved → {config.RF_MODEL_PATH}")

    # Global evaluation
    test_preds  = best_model.predict(X_test)
    test_metrics = _metrics(y_test, test_preds, "Test Set")
    baseline_mse = test_metrics["MSE"]

    # Granular evaluation
    test_df = test.copy()
    test_df["predicted_volume_m"] = test_preds

    print("\n--- By TPH ---")
    for tph_val in sorted(test_df["tph"].unique()):
        sub = test_df[test_df["tph"] == tph_val]
        _metrics(sub[config.TARGET_COL], sub["predicted_volume_m"], f"TPH {tph_val}")

    print("\n--- By Age ---")
    for age_val in sorted(test_df["age"].unique()):
        sub = test_df[test_df["age"] == age_val]
        _metrics(sub[config.TARGET_COL], sub["predicted_volume_m"], f"Age {age_val}")

    # PFI
    print("\n--- Overall Permutation Feature Importance ---")
    compute_pfi_overall(best_model, X_test, y_test, baseline_mse)

    print("\n--- PFI by Age Group ---")
    compute_pfi_by_age(best_model, test, X_test, y_test)

    # Save predictions
    save_predictions(test, test_preds)

    print("\nStage 3 complete.\n")


if __name__ == "__main__":
    run()
