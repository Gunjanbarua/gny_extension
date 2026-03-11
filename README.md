# Tree Volume Prediction Pipeline

A reproducible machine-learning pipeline for predicting individual tree
volume in spacing-trial plantations using LiDAR-derived stand structure
and competition-index features.

This project compares three modelling approaches — **Random Forest (RF)**,
**Support Vector Regression (SVR)**, and a **Linear Mixed-Effects Model (LMM)** —
and investigates how predictor importance varies across stand age and planting
density (trees-per-hectare, TPH).

> **Data**: The raw field measurements and LiDAR-derived features used in
> this study are not included in this repository.  Placeholder directories
> (`data/`, `models/`, `results/`, `figures/`) show where outputs are expected.

---

## Study Context

The data come from the **RW20 spacing trial**, a long-term forestry experiment
with four sites (RH, BL, STP) spanning planting densities of 618, 1 236, and
1 853 trees per hectare.  Repeated measurements of diameter at breast height
(DBH) and tree height were collected annually from stand age 9 to 16 years.

LiDAR point-cloud processing yielded per-tree crown metrics (crown area,
stem volume proxies, stem-form attributes) and competition indices (CI)
that serve as predictors.  The target variable is individual-tree
merchantable volume in m³ (Honer's equation; Honer 1967, doi:10.1093/sjaf/21.3.146).

---

## Repository Layout

```
gny_extension/
├── config.py               # Central configuration (paths, hyperparameters)
├── run_pipeline.py         # Top-level entry point
├── requirements.txt
│
├── src/
│   ├── data_preparation.py # Stage 1 – cleaning, imputation, feature engineering
│   ├── lmm.py              # Stage 2 – Lasso selection + Linear Mixed-Effects Model
│   ├── random_forest.py    # Stage 3 – RF with spatial CV, tuning, and PFI
│   ├── pfi_analysis.py     # Stage 4 – Permutation Feature Importance by TPH
│   └── visualization.py    # Stage 5 – all figures (growth curves, scatter, PFI)
│
├── data/                   # ← place input CSVs here (not tracked)
├── models/                 # ← saved joblib model files (not tracked)
├── results/                # ← prediction and PFI CSVs (not tracked)
└── figures/                # ← output figures (not tracked)
```

---

## Pipeline Stages

### Stage 1 – Data Preparation (`src/data_preparation.py`)

| Step | Description |
|------|-------------|
| Height imputation | A quadratic DBH → Height model is fitted on complete cases; missing heights are imputed from the model. |
| Feature derivation | TPH label, site code, composite tree ID (`unique_id`), volume (ft³ and m³). |
| Dataset merge | Per-tree field data is left-joined onto the LiDAR/CI master sheet on `unique_id`. The Nelder-wheel treatment (non-rectangular spacing) is excluded. |

**Input files**
- `data/rw20_field_data.csv` – raw field measurements (DBH, height, plot, tree)
- `data/master_datasheet_rf.csv` – LiDAR crown metrics and competition indices

**Output**
- `data/input_data.csv` – final modelling dataset (~104k rows × 49 columns)

---

### Stage 2 – Linear Mixed-Effects Model (`src/lmm.py`)

A classical statistical baseline that accounts for the repeated-measures
structure of the data (multiple annual measurements per tree).

| Step | Description |
|------|-------------|
| Train/test split | 80/20 split at the tree level (`unique_id`) so all measurements of a tree are in one partition. |
| Variable selection | Lasso regression (α=0.01, standardised features) selects a sparse subset of the 38 LiDAR/CI candidates. |
| VIF check | Variance Inflation Factors confirm the selected features are not harmfully collinear. |
| Model | `volume_m ~ age + <selected features>` with random intercepts and random slopes on `age` per tree, fitted by REML (L-BFGS). |
| Diagnostics | Breusch-Pagan heteroscedasticity test, residuals-vs-fitted plot, Q-Q plot. |

**Key results (test set)**
- R² ≈ 0.17 · nRMSE ≈ 53% — the LMM serves as a statistical reference, not a competitive predictor.

---

### Stage 3 – Random Forest (`src/random_forest.py`)

The primary predictive model.

| Step | Description |
|------|-------------|
| Train/test split | 80/20 stratified by study site, split at the **plot** level to preserve spatial independence. |
| Hyperparameter tuning | `HalvingGridSearchCV` over 432 candidate configurations, with 5-fold `GroupKFold` CV (groups = plots). |
| Best parameters | `n_estimators=400`, `max_features=0.5`, `max_depth=None`, `min_samples_leaf=2`, `min_samples_split=5` |
| Evaluation | Global metrics plus stratification by TPH group and stand age. |
| Permutation Feature Importance | Computed overall (50 repeats) and per age class (200 repeats). |

**Key results (test set)**
- R² ≈ 0.90 · nRMSE ≈ 18% · Mean Bias ≈ –0.001 m³

Top predictors (overall): `age`, `vol5`, `CI_Z`, `Z`, `Carea`, `sfa5`

---

### Stage 4 – PFI by Planting Density (`src/pfi_analysis.py`)

Repeats the Permutation Feature Importance calculation separately for
each TPH group (618, 1 236, 1 853 trees ha⁻¹) using both the RF and
SVR models.

This reveals whether the relative importance of structural and
competition-index features shifts with stand density — a key
silvicultural question for operational forest management.

**Finding**: `age` dominates at low density (618 tph, 873% MSE increase)
but its relative importance decreases as competition intensifies, while
`CI_Z` (height-based competition index) rises in importance.

---

### Stage 5 – Visualisation (`src/visualization.py`)

| Figure | Description |
|--------|-------------|
| `comparison_growth_curves.jpg` | Mean predicted vs field volume ± IQR across age for RF and SVR |
| `rf_svr_scatter_tph.jpg` | Predicted vs field scatter coloured by age, one panel per TPH × model |
| `rf_svr_residuals_tph.jpg` | Residuals vs field volume by TPH × model |
| `overlapping_hist.jpg` | Actual vs predicted volume distributions (histogram + KDE) |
| `nrmse_by_age_rf_svr.jpg` | nRMSE trajectory across age classes |
| `model_nrmse_tph_age.jpg` | nRMSE faceted by model, coloured by TPH |
| `pfi_overall_rf_svr.jpg` | Overall PFI bar chart for RF and SVR |
| `pfi_rf_age.jpg` / `pfi_svr_age.jpg` | PFI trend lines across age groups |
| `pfi_tph_rf_svr.jpg` | PFI by TPH for RF and SVR side-by-side |

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files in data/ (see Stage 1 for required filenames)

# 4. Run the full pipeline
python run_pipeline.py

# 5. Run only specific stages
python run_pipeline.py --stages 3,4,5

# 6. Resume from a stage (e.g. after Stage 3 is complete)
python run_pipeline.py --from-stage 4
```

---

## Design Decisions

**Spatial train/test split** (Stage 3):
Splitting at the plot level (rather than the tree or row level) prevents
spatial autocorrelation from leaking into the test set, giving a more
honest estimate of generalisation performance.

**Group-based CV** (Stage 3):
`GroupKFold` ensures that the same plot never appears in both the
training and validation fold during hyperparameter search.

**Repeated-measures awareness** (Stage 2):
The LMM explicitly models the correlation structure of annual measurements
on the same tree via random slopes on `age`, which plain OLS ignores.

**Permutation over impurity-based importance** (Stages 3 & 4):
PFI is computed on held-out data, making it robust to the bias that
impurity-based (Gini) importance exhibits toward high-cardinality features.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas / numpy | Data manipulation |
| scipy | Curve fitting (height model) |
| scikit-learn | RF, SVR, PFI, model selection |
| statsmodels | LMM, VIF, Breusch-Pagan test |
| matplotlib / seaborn | Visualisation |
| joblib | Model serialisation |
