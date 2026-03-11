"""
src/data_preparation.py
-----------------------
Pipeline Stage 1 – Data cleaning and feature engineering.

Steps
-----
1. Load raw field measurements (DBH, height, plot metadata).
2. Fit a quadratic DBH → Height model and impute any missing heights.
3. Derive tree-level attributes: volume, unique_id, TPH, study labels.
4. Load the LiDAR / competition-index master datasheet and clean it.
5. Merge the two datasets on unique_id, drop the 'NLD' study, and save
   the final modelling dataset to ``data/input_data.csv``.

Source notebook: 1.data_organizing.ipynb
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Helper: quadratic height–DBH model
# ---------------------------------------------------------------------------

def _quadratic(dbh, a, b, c):
    """H = a + b·DBH + c·DBH²  (Chapman-Richards style polynomial)."""
    return a + b * dbh + c * dbh ** 2


def fit_height_model(df: pd.DataFrame, dbh_col: str = "dbh", ht_col: str = "ht"):
    """
    Fit a quadratic DBH→Height model on rows that have both measurements.

    Returns
    -------
    popt : array-like
        Fitted coefficients (a, b, c).
    metrics : dict
        R² and MAE of the fitted model on training data.

    Reference: USDA GTR-NRS-200 (https://www.fs.usda.gov/nrs/pubs/gtr/gtr_nrs200.pdf)
    """
    complete = df.dropna(subset=[dbh_col, ht_col])
    X = complete[dbh_col].values
    y = complete[ht_col].values

    popt, _ = curve_fit(_quadratic, X, y)
    a, b, c = popt

    y_pred = _quadratic(X, a, b, c)
    metrics = {
        "r2":  r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
    }

    print(
        f"Height model: H = {a:.4f} + {b:.4f}·DBH + {c:.4f}·DBH²  "
        f"(R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f} ft)"
    )
    return popt, metrics


def impute_heights(df: pd.DataFrame, popt, dbh_col: str = "dbh", ht_col: str = "ht"):
    """Fill rows where height is missing but DBH is recorded."""
    mask = df[ht_col].isnull() & df[dbh_col].notnull()
    df.loc[mask, ht_col] = _quadratic(df.loc[mask, dbh_col], *popt)
    print(f"Imputed {mask.sum()} missing height values.")
    return df


# ---------------------------------------------------------------------------
# Stage 1a – Clean raw field data
# ---------------------------------------------------------------------------

def prepare_field_data(
    raw_path=config.RAW_FIELD_DATA,
    output_path=config.CLEAN_FIELD_DATA,
) -> pd.DataFrame:
    """
    Load, clean, and enrich the raw per-tree field measurements.

    Derived columns
    ---------------
    tph        : Trees-per-hectare label from spacing treatment code.
    study      : Human-readable site label (RH, BL, STP, NLD).
    unique_id  : Composite tree identifier '<study>-<plot>-<tree>'.
    volume     : Merchantable volume (ft³) via Honer's formula.
    volume_m   : Volume converted to m³.
    age        : Measurement year-since-establishment ('yst' column).

    Reference for volume formula: doi:10.1093/sjaf/21.3.146
    """
    df = pd.read_csv(raw_path, low_memory=False)
    df = df.dropna(subset=["dbh", "ht"])

    # Impute missing heights
    popt, _ = fit_height_model(df)
    df = impute_heights(df, popt)

    # Map spacing code → trees-per-hectare
    df["tph"] = df["space"].replace(config.TPH_MAP)

    # Map numeric study code → site abbreviation
    study_map_inv = {v: k for k, v in config.STUDY_MAP.items()}  # e.g. 'RH' → 201301
    df["study"] = df["study"].replace(
        {201301: "RH", 201302: "BL", 201303: "STP", 201304: "NLD"}
    )

    # Composite tree ID
    df["unique_id"] = (
        df["study"].astype(str) + "-"
        + df["plot"].astype(str) + "-"
        + df["tree"].astype(str)
    )

    # Volume: Honer's merchantable volume equation (ft³ → m³)
    df["volume"]   = 0.21949 + 0.00238 * (df["dbh"] ** 2) * df["ht"]
    df["volume_m"] = df["volume"] / 35.315

    df["age"] = df["yst"]

    output_cols = ["study", "plot", "tree", "unique_id", "age", "tph",
                   "dbh", "ht", "volume", "volume_m"]
    df_out = df[df["dbh"].notnull()][output_cols].copy()

    df_out.to_csv(output_path, index=False)
    print(f"Clean field data saved → {output_path}  shape={df_out.shape}")
    return df_out


# ---------------------------------------------------------------------------
# Stage 1b – Clean master LiDAR / competition-index datasheet
# ---------------------------------------------------------------------------

def prepare_master_datasheet(
    raw_path=config.MASTER_DATASHEET,
    output_path=config.CLEAN_MASTER,
) -> pd.DataFrame:
    """
    Load and clean the LiDAR-derived stand structure and competition index
    master datasheet, then add a composite tree identifier.
    """
    df = pd.read_csv(raw_path, low_memory=False)

    # Drop columns not needed downstream
    drop_cols = [c for c in ["unique", "volume_feet_21"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Composite tree ID (matches field data)
    df["unique_id"] = (
        df["study"].astype(str) + "-"
        + df["plot.x"].astype(str) + "-"
        + df["treeID"].astype(str)
    )

    df.to_csv(output_path, index=False)
    print(f"Clean master datasheet saved → {output_path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Stage 1c – Merge and produce modelling dataset
# ---------------------------------------------------------------------------

def merge_datasets(
    field_path=config.CLEAN_FIELD_DATA,
    master_path=config.CLEAN_MASTER,
    output_path=config.INPUT_DATA,
) -> pd.DataFrame:
    """
    Left-join field measurements onto LiDAR features using unique_id.

    The 'NLD' (Nelder) study is excluded because its non-standard
    planting geometry makes it incomparable to the rectangular spacing
    treatments used for modelling.
    """
    df_field  = pd.read_csv(field_path,  low_memory=False)
    df_master = pd.read_csv(master_path, low_memory=False)

    # Drop raw structural columns already encoded in volume_m
    df_field = df_field.drop(columns=["dbh", "ht", "volume"], errors="ignore")

    merged = pd.merge(df_field, df_master, on="unique_id", how="left")

    # Exclude Nelder-wheel study
    merged = merged[merged["study_x"] != "NLD"]
    merged = merged.dropna()

    merged.to_csv(output_path, index=False)
    print(
        f"Merged dataset saved → {output_path}\n"
        f"  Field rows:  {len(df_field):>8,}\n"
        f"  Master rows: {len(df_master):>8,}\n"
        f"  Final shape: {merged.shape}"
    )
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run():
    """Execute all three preparation stages in order."""
    print("\n" + "=" * 60)
    print("STAGE 1 – Data Preparation")
    print("=" * 60)
    prepare_field_data()
    prepare_master_datasheet()
    merge_datasets()
    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    run()
