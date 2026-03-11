"""
config.py
---------
Central configuration for the tree volume prediction pipeline.
All file paths, column definitions, and model hyperparameters are
defined here so that every module stays in sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
RESULTS_DIR  = ROOT_DIR / "results"
FIGURES_DIR  = ROOT_DIR / "figures"

# ---------------------------------------------------------------------------
# Raw / intermediate data files
# ---------------------------------------------------------------------------
RAW_FIELD_DATA      = DATA_DIR / "rw20_field_data.csv"
MASTER_DATASHEET    = DATA_DIR / "master_datasheet_rf.csv"
CLEAN_FIELD_DATA    = DATA_DIR / "rw20_field_data_clean.csv"
CLEAN_MASTER        = DATA_DIR / "master_datasheet_rf_clean.csv"
INPUT_DATA          = DATA_DIR / "input_data.csv"

# ---------------------------------------------------------------------------
# Result files produced by each model
# ---------------------------------------------------------------------------
RF_PREDICTIONS      = RESULTS_DIR / "rf_final.csv"
LMM_PREDICTIONS     = RESULTS_DIR / "linear_mixed_model2.csv"

PFI_RF_OVERALL      = RESULTS_DIR / "pfi_rf_overall_final.csv"
PFI_RF_AGE          = RESULTS_DIR / "pfi_rf_age_final.csv"
PFI_RF_TPH          = RESULTS_DIR / "pfi_rf_tph_final.csv"
PFI_SVR_TPH         = RESULTS_DIR / "pfi_svr_tph_final.csv"

# ---------------------------------------------------------------------------
# Saved model artefacts
# ---------------------------------------------------------------------------
RF_MODEL_PATH       = MODELS_DIR / "rf_final.joblib"
SVR_MODEL_PATH      = MODELS_DIR / "svr_final.joblib"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
TARGET_COL      = "volume_m"
GROUP_COL       = "unique_id"          # tree-level repeated-measures ID
RANDOM_SLOPE    = "age"
PLOT_COL        = "plot"
STUDY_COL       = "study_x"

# Columns dropped before modelling (identifiers, targets, leakage)
COLS_TO_DROP = [
    "study_x", "plot", "tree", "unique_id", "tph",
    "treeID", "study_y", "plot.x", "study_code", TARGET_COL,
]

# All candidate fixed-effect features evaluated by Lasso / RF
FEATURE_CANDIDATES = [
    "Carea", "mCDst", "Z", "CArea_1", "HTLC.x", "CLAI",
    "vol1", "vol2", "vol3", "vol4", "vol5",
    "sfa1", "sfa2", "sfa3", "sfa4", "sfa5",
    "UndTF", "UndPrp",
    "CI_Carea", "CI_CArea_1", "CI_Z", "CI_mCDst", "CI_LAI", "CI_HTLC",
    "CI_under", "CI_under2",
    "CI_vol1", "CI_vol2", "CI_vol3", "CI_vol4", "CI_vol5",
    "CI_sfa1", "CI_sfa2", "CI_sfa3", "CI_sfa4", "CI_sfa5",
    "SILVA1", "SILVA2",
]

# Planting density (trees-per-hectare) codes and their integer values
TPH_MAP = {1: 618, 2: 1236, 3: 1853, 4: "Nelders"}

# Study site code mappings
STUDY_MAP = {201301: "RH", 201302: "BL", 201303: "STP", 201304: "NLD"}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.20
RANDOM_STATE = 200          # used consistently for all splits & models

# ---------------------------------------------------------------------------
# Random-Forest hyperparameter grid (HalvingGridSearchCV)
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 400],
    "max_features":      ["sqrt", "log2", 0.5],
    "max_depth":         [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
RF_CV_FOLDS  = 5
RF_CV_FACTOR = 2            # halving factor

# ---------------------------------------------------------------------------
# Lasso (for LMM variable selection)
# ---------------------------------------------------------------------------
LASSO_ALPHA      = 0.01
LASSO_MAX_ITER   = 10_000
LASSO_RANDOM     = 300

# ---------------------------------------------------------------------------
# Permutation Feature Importance
# ---------------------------------------------------------------------------
PFI_N_REPEATS_OVERALL = 50
PFI_N_REPEATS_AGE     = 200
PFI_N_REPEATS_TPH     = 30
