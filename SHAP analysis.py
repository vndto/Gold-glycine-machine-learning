"""
SHAP feature-attribution analysis: Random Forest vs XGBoost
Data:     Beaker_Only AND Microfluidic_Only sheets (run separately)
Target:   Au_extraction_%

Rules this script follows:
  1. Beaker and Microfluidic are NEVER pooled. Each dataset gets its own
     train/test split, its own tuned model, and its own SHAP explainer whose
     background AND explanation set are both that dataset's own held-out
     test split. The two datasets have different feature sets and the
     target lives on different scales (Beaker ~0-1.3, Microfluidic ~0-100),
     so a shared background would mix incomparable distributions.
  1b. The split is a plain random row-wise train_test_split (test_size=0.2,
      random_state=42) with KFold(shuffle=True) for tuning — matching how
      the models are actually trained elsewhere in this project. Rows from
      the same kinetic run can land on both sides of the split, so Time_h's
      SHAP rank should be read with that in mind.
  2. SHAP values are never averaged across datasets or across models — the
     units/scales differ between XGBoost and Random Forest and between the
     two datasets. Only the RANK ORDER of features is compared, across all
     four runs (RF/XGB x Beaker/Microfluidic) at once, via pairwise
     Kendall's tau — never by combining magnitudes.
  2b. Time_h stays in every model and in each run's own SHAP outputs
      (beeswarm/bar plots, per-run importance CSV) — it is a real
      predictor and is fitted and explained like any other feature. It is
      excluded ONLY from the cross-run rank/tau summary below, since that
      table is about comparing agreement on the process-condition
      features, not about Time_h itself.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from itertools import combinations
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(OUTDIR, "Preprocessed data.xlsx")
TARGET = "Au_extraction_%"

BEAKER_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH",
]

MICRO_FEATURES = [
    "Glycine_M", "Time_h", "H2O2_%",
    "Temp_C", "pH",
]

# Excluded only from the cross-run rank/tau summary at the bottom of this
# script (see rule 2b above) — it stays in the model and in every run's
# own SHAP outputs.
RANK_SUMMARY_EXCLUDE = {"Time_h"}

DATASETS = {
    "beaker":       dict(sheet="Beaker_Only",      features=BEAKER_FEATURES),
    "microfluidic": dict(sheet="Microfluidic_Only", features=MICRO_FEATURES),
}

# Same tuning grids as RF.py / XGBoost.py, so the "best" model explained
# here is the same model those scripts report metrics for.
RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 500],
    "max_depth":         [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 1.0],
}

XGB_PARAM_GRID = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3],
}


# -------------------------------------------------------------------
# Model fitting
# -------------------------------------------------------------------
def fit_rf(X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=RF_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def fit_xgb(X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, verbosity=0, tree_method="hist",
    )
    grid = GridSearchCV(
        base, param_grid=XGB_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


MODEL_FITTERS = {"rf": fit_rf, "xgb": fit_xgb}


# -------------------------------------------------------------------
# SHAP for one dataset
# -------------------------------------------------------------------
def run_shap(tag, sheet, features):
    print("\n" + "=" * 70)
    print(f"  SHAP — {tag}  ({sheet})")
    print("=" * 70)

    df = pd.read_excel(FILE, sheet_name=sheet)

    # Verify variance directly instead of reading it off the beeswarm plot.
    # nunique <= 1 -> truly constant, no split is possible, drop it.
    # nunique == 2 -> a two-level variable: SHAP can still show which
    # direction it pushes the prediction, but with only two settings tested
    # its rank against continuous features isn't a meaningful comparison,
    # so it stays in the model and gets flagged in the importance table.
    nunique = df[features].nunique(dropna=False)
    constant_features = nunique[nunique <= 1].index.tolist()
    two_level_features = nunique[nunique == 2].index.tolist()
    if constant_features:
        print(f"[{tag}] dropping constant feature(s) (nunique<=1): {constant_features}")
        features = [f for f in features if f not in constant_features]
    if two_level_features:
        print(f"[{tag}] two-level feature(s) (nunique==2), kept in model but "
              f"flagged in the importance table: {two_level_features}")

    X = df[features]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    model_results = {}
    for model_tag, fitter in MODEL_FITTERS.items():
        print(f"\n--- Tuning {model_tag.upper()} on {tag} ---")
        model = fitter(X_train, y_train)

        # Background AND explanation set are both this dataset's own
        # held-out test split — never shared with the other dataset.
        explainer = shap.Explainer(model, X_test)
        explanation = explainer(X_test)

        mean_abs = np.abs(explanation.values).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature":       features,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1
        imp_df["note"] = imp_df["feature"].apply(
            lambda f: "two-level feature: direction only, rank not reliable"
            if f in two_level_features else "")

        print(f"\n=== SHAP importance — {model_tag.upper()} — {tag} ===")
        print(imp_df.to_string(index=False))

        csv_name = os.path.join(OUTDIR, f"shap_importance_{model_tag}_{tag}.csv")
        imp_df.to_csv(csv_name, index=False)
        print(f"Saved: {csv_name}")

        plt.figure()
        shap.plots.beeswarm(explanation, show=False)
        plt.title(f"SHAP summary — {model_tag.upper()} — {tag}")
        plt.tight_layout()
        fname = os.path.join(OUTDIR, f"shap_beeswarm_{model_tag}_{tag}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

        plt.figure()
        shap.plots.bar(explanation, show=False)
        plt.title(f"Mean |SHAP| — {model_tag.upper()} — {tag}")
        plt.tight_layout()
        fname = os.path.join(OUTDIR, f"shap_bar_{model_tag}_{tag}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

        model_results[model_tag] = imp_df.set_index("feature")

    return model_results


# -------------------------------------------------------------------
# Run each dataset independently
# -------------------------------------------------------------------
all_results = {}
for tag, cfg in DATASETS.items():
    all_results[tag] = run_shap(tag, cfg["sheet"], cfg["features"])

# -------------------------------------------------------------------
# Compact cross-run summary: feature rank in each of the four runs, plus
# pairwise Kendall's tau between every pair of runs. Ranks (not SHAP
# magnitudes) are what's compared. Time_h is excluded here (rule 2b) and
# ranks are recomputed over the remaining features only, so removing it
# doesn't leave gaps in the 1..N ranking. Beaker and Microfluidic share
# the same feature set here, but the reindex/dropna below stay in place in
# case that ever changes (a feature unique to one dataset would show as
# NaN in the other's columns, and tau for that pair is computed on their
# shared, non-NaN features only).
# -------------------------------------------------------------------
RUNS = [
    ("rf",  "beaker",       "RF – Beaker"),
    ("xgb", "beaker",       "XGB – Beaker"),
    ("rf",  "microfluidic", "RF – Microfluidic"),
    ("xgb", "microfluidic", "XGB – Microfluidic"),
]

RANK_FEATURES = [f for f in BEAKER_FEATURES if f not in RANK_SUMMARY_EXCLUDE]

rank_table = pd.DataFrame(index=RANK_FEATURES)
for model_tag, tag, label in RUNS:
    shap_vals = all_results[tag][model_tag]["mean_abs_shap"].reindex(RANK_FEATURES)
    rank_table[label] = shap_vals.rank(ascending=False)

print("\n" + "=" * 70)
print("  Feature rank summary — all four runs")
print("=" * 70)
print(rank_table.to_string(float_format=lambda v: f"{v:.0f}"))

rank_csv = os.path.join(OUTDIR, "shap_rank_summary.csv")
rank_table.to_csv(rank_csv)
print(f"Saved: {rank_csv}")

labels = [label for _, _, label in RUNS]
tau_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
for a, b in combinations(labels, 2):
    pair = rank_table[[a, b]].dropna()
    tau, pval = kendalltau(pair[a], pair[b])
    tau_matrix.loc[a, b] = tau_matrix.loc[b, a] = tau

print("\n" + "=" * 70)
print("  Pairwise rank agreement — Kendall's tau")
print("=" * 70)
print(tau_matrix.to_string(float_format=lambda v: f"{v:.3f}"))

tau_csv = os.path.join(OUTDIR, "shap_rank_tau_matrix.csv")
tau_matrix.to_csv(tau_csv)
print(f"Saved: {tau_csv}")

# Mean tau of each run against the other three — a run that agrees most
# with the rest of the field is a reasonable "representative" choice for
# the one beeswarm shown in the main text (the other three go in the SI).
mean_tau = tau_matrix.mean(axis=1, skipna=True).sort_values(ascending=False)
print("\nMean tau vs. the other three runs (highest = most representative):")
print(mean_tau.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\nSuggested main-text beeswarm: {mean_tau.index[0]}"
      f" (mean tau = {mean_tau.iloc[0]:.3f})")
