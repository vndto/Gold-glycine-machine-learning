"""
Feature importance analysis for Gaussian Process Regression
Data:     Beaker_Only AND Microfluidic_Only sheets
Targets:  Beaker      -> Au_dissolution_ppm
          Microfluidic -> Au_extraction_%

Method: Permutation importance — measures the drop in R² when each
        feature is randomly shuffled. This is model-agnostic and works
        for GPR (which has no built-in feature_importances_).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C, WhiteKernel,
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(OUTDIR, "Preprocessed data.xlsx")

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

def make_kernels():
    return [
        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2)),
        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2)),
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2)),
    ]

PARAM_GRID = {
    "kernel": make_kernels(),
    "alpha":  [1e-10, 1e-6, 1e-4, 1e-2],
    "normalize_y": [True],
}

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def viridis_gradient(n):
    return plt.cm.viridis(np.linspace(0.15, 0.95, n))


def run_importance(sheet, features, target, tag):
    print("\n" + "=" * 60)
    print(f"  Feature importance (GPR) — {sheet}")
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
              f"flagged in the importance table: {two_level_features}")

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Fit tuned GPR
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        GaussianProcessRegressor(n_restarts_optimizer=5, random_state=42),
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    print("\nTuning GPR ...")
    grid.fit(X_train, y_train)
    best_gpr = grid.best_estimator_
    print(f"Best kernel: {best_gpr.kernel_}")
    print(f"Test R² : {r2_score(y_test, best_gpr.predict(X_test)):.4f}")

    # Permutation importance on TEST set
    # (better than train set — reflects real generalization impact)
    print("\nComputing permutation importance ...")
    result = permutation_importance(
        best_gpr, X_test, y_test,
        n_repeats=30,
        random_state=42,
        scoring="r2",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=True)  # ascending for horizontal bar plot
    importance_df["note"] = importance_df["feature"].apply(
        lambda f: "two-level feature: direction only, rank not reliable"
        if f in two_level_features else "")

    print("\n=== Permutation importance (R² drop when shuffled) ===")
    print(importance_df.sort_values("importance", ascending=False).to_string(index=False))

    # Horizontal bar plot with error bars
    plt.figure(figsize=(8, 6))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
        xerr=importance_df["std"],
        color=viridis_gradient(len(importance_df)),
        ecolor="gray",
        capsize=3,
    )
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("Permutation importance (mean R² drop)")
    plt.title(f"Feature importance — {sheet}\n(target: {target})")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fname = os.path.join(OUTDIR, f"gpr_importance_{tag}.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

    # Save to CSV
    csv_name = os.path.join(OUTDIR, f"gpr_importance_{tag}.csv")
    importance_df.sort_values("importance", ascending=False).to_csv(
        csv_name, index=False
    )
    print(f"Saved: {csv_name}")

    return importance_df


# -------------------------------------------------------------------
# Run on both sheets
# -------------------------------------------------------------------
imp_beaker = run_importance(
    "Beaker_Only", BEAKER_FEATURES, "Au_extraction_%", "beaker"
)
imp_micro = run_importance(
    "Microfluidic_Only", MICRO_FEATURES, "Au_extraction_%", "microfluidic"
)

# -------------------------------------------------------------------
# Side-by-side comparison plot
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

for ax, df_imp, title in [
    (axes[0], imp_beaker, "Beaker  (Au_extraction_%)"),
    (axes[1], imp_micro,  "Microfluidic  (Au_extraction_%)"),
]:
    ax.barh(df_imp["feature"], df_imp["importance"],
            xerr=df_imp["std"], color=viridis_gradient(len(df_imp)),
            ecolor="gray", capsize=3)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Permutation importance (mean R² drop)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")

fig.suptitle("GPR permutation feature importance — Beaker vs Microfluidic",
             fontsize=13, y=1.02)
plt.tight_layout()
comparison_fname = os.path.join(OUTDIR, "gpr_importance_comparison.png")
plt.savefig(comparison_fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {comparison_fname}")