"""
Feature importance analysis for XGBoost
Data:     Beaker_Only AND Microfluidic_Only sheets
Targets:  Beaker       -> Au_dissolution_ppm
          Microfluidic -> Au_extraction_%

Two importance methods are computed and plotted side by side:
  1. Built-in gain-based importance  (how much each feature improves
     the objective across all splits)
  2. Permutation importance on the TEST set  (drop in R² when the
     feature is shuffled — model-agnostic, reflects generalization)

Requires: pip install xgboost pandas scikit-learn matplotlib openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FILE = "Preprocessed data.xlsx"

BEAKER_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH",
]

# Microfluidic candidates; zero-variance columns (Cyanide_ppm, Cu_ppm,
# KMnO4_M, Agitation_rpm, ... are all constant = 0 in this sheet) are
# verified and dropped dynamically at runtime — see the nunique check in
# run_importance() — rather than hardcoded here from visual inspection.
MICRO_FEATURES = [
    "Glycine_M", "Time_h", "H2O2_%",
    "Temp_C", "pH",
]

PARAM_GRID = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3],
}

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def viridis_gradient(n):
    return plt.cm.viridis(np.linspace(0.15, 0.95, n))


def run_importance(sheet, features, target, tag):
    print("\n" + "=" * 60)
    print(f"  Feature importance (XGBoost) — {sheet}")
    print(f"  Target: {target}")
    print("=" * 60)

    df = pd.read_excel(FILE, sheet_name=sheet)

    # Verify variance directly (nunique) rather than assuming from a
    # hardcoded list: nunique<=1 -> constant, no split possible, drop it;
    # nunique==2 -> a two-level variable that supports a direction but not
    # a magnitude/importance ranking against continuous features.
    nunique = df[features].nunique(dropna=False)
    constant_features = nunique[nunique <= 1].index.tolist()
    two_level_features = nunique[nunique == 2].index.tolist()
    if constant_features:
        print(f"[{tag}] dropping constant feature(s) (nunique<=1): {constant_features}")
        features = [f for f in features if f not in constant_features]
    if two_level_features:
        print(f"[{tag}] two-level feature(s) (nunique==2), kept in model but "
              f"flagged in the importance tables: {two_level_features}")

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Fit tuned XGBoost
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )

    grid = GridSearchCV(
        estimator  = base_model,
        param_grid = PARAM_GRID,
        scoring    = "neg_root_mean_squared_error",
        cv         = cv,
        n_jobs     = -1,
        verbose    = 1,
    )
    print("\nTuning XGBoost ...")
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    print(f"Test R² : {r2_score(y_test, best_xgb.predict(X_test)):.4f}")

    # ---------------------------------------------------------------
    # 1. Built-in gain-based importance
    # ---------------------------------------------------------------
    gain_df = pd.DataFrame({
        "feature":    features,
        "importance": best_xgb.feature_importances_,
    }).sort_values("importance", ascending=True)
    gain_df["note"] = gain_df["feature"].apply(
        lambda f: "two-level feature: direction only, rank not reliable"
        if f in two_level_features else "")

    print("\n=== Gain-based importance ===")
    print(gain_df.sort_values("importance", ascending=False).to_string(index=False))

    # ---------------------------------------------------------------
    # 2. Permutation importance on TEST set
    # ---------------------------------------------------------------
    print("\nComputing permutation importance ...")
    result = permutation_importance(
        best_xgb, X_test, y_test,
        n_repeats=30,
        random_state=42,
        scoring="r2",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature":    features,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=True)
    perm_df["note"] = perm_df["feature"].apply(
        lambda f: "two-level feature: direction only, rank not reliable"
        if f in two_level_features else "")

    print("\n=== Permutation importance (R² drop when shuffled) ===")
    print(perm_df.sort_values("importance", ascending=False).to_string(index=False))

    # ---------------------------------------------------------------
    # Side-by-side plot: gain vs permutation
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].barh(gain_df["feature"], gain_df["importance"],
                 color=viridis_gradient(len(gain_df)))
    axes[0].set_xlabel("Gain-based importance")
    axes[0].set_title("Built-in (gain)")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(perm_df["feature"], perm_df["importance"],
                 xerr=perm_df["std"], color=viridis_gradient(len(perm_df)),
                 ecolor="gray", capsize=3)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Permutation importance (mean R² drop)")
    axes[1].set_title("Permutation (test set)")
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle(f"XGBoost feature importance — {sheet}\n(target: {target})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f"xgb_importance_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

    # Save to CSV
    gain_df.sort_values("importance", ascending=False).to_csv(
        f"xgb_gain_importance_{tag}.csv", index=False)
    perm_df.sort_values("importance", ascending=False).to_csv(
        f"xgb_perm_importance_{tag}.csv", index=False)
    print(f"Saved CSVs: xgb_gain_importance_{tag}.csv, xgb_perm_importance_{tag}.csv")

    return gain_df, perm_df


# -------------------------------------------------------------------
# Run on both sheets
# -------------------------------------------------------------------
gain_b, perm_b = run_importance(
    "Beaker_Only", BEAKER_FEATURES, "Au_extraction_%", "beaker"
)
gain_m, perm_m = run_importance(
    "Microfluidic_Only", MICRO_FEATURES, "Au_extraction_%", "microfluidic"
)

# -------------------------------------------------------------------
# Cross-system comparison (permutation importance only)
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df_imp, title in [
    (axes[0], perm_b, "Beaker  (Au_extraction_%)"),
    (axes[1], perm_m, "Microfluidic  (Au_extraction_%)"),
]:
    ax.barh(df_imp["feature"], df_imp["importance"],
            xerr=df_imp["std"], color=viridis_gradient(len(df_imp)),
            ecolor="gray", capsize=3)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Permutation importance (mean R² drop)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")

fig.suptitle("XGBoost permutation feature importance — Beaker vs Microfluidic",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("xgb_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: xgb_importance_comparison.png")