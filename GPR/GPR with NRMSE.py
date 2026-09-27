"""
Machine learning study: Gold leaching with glycine
Model:    Gaussian Process Regression with hyperparameter tuning
Data:     Beaker_Only AND Microfluidic_Only sheets (run separately)
Targets:  Beaker       -> Au_extraction_%
          Microfluidic -> Au_extraction_%
Split:    80% train / 20% test

Metrics: RMSE, MAE, MAPE, R², NRMSE_range, NRMSE_mean
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(OUTDIR, "Preprocessed data.xlsx")

BEAKER_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH",
]

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
# Evaluation helper
# -------------------------------------------------------------------
def evaluate(y_true, y_pred, label, y_full):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any() else np.nan)

    y_range = y_full.max() - y_full.min()
    y_mean  = y_full.mean()
    nrmse_range = rmse / y_range if y_range > 0 else np.nan
    nrmse_mean  = rmse / y_mean  if y_mean  != 0 else np.nan

    print(f"\n=== {label} performance ===")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  MAPE          : {mape:.2f}%  (non-zero targets only)")
    print(f"  R^2           : {r2:.4f}")
    print(f"  NRMSE (range) : {nrmse_range:.4f}  ({nrmse_range*100:.2f}% of range)")
    print(f"  NRMSE (mean)  : {nrmse_mean:.4f}  ({nrmse_mean*100:.2f}% of mean)")
    return rmse, mae, mape, r2, nrmse_range, nrmse_mean


def run_gpr(sheet, features, target, tag, lims):
    print("\n" + "=" * 60)
    print(f"  GPR — {sheet}  (target: {target})")
    print("=" * 60)

    df = pd.read_excel(FILE, sheet_name=sheet)
    print(f"Loaded {sheet}: {df.shape[0]} rows")

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
        print(f"[{tag}] two-level feature(s) (nunique==2), kept in model: {two_level_features}")

    X = df[features].values
    y = df[target].values
    print(f"Target: {target} | range [{y.min():.4f}, {y.max():.4f}] | mean {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        GaussianProcessRegressor(n_restarts_optimizer=5, random_state=42),
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    print("\nRunning grid search ...")
    grid.fit(X_train, y_train)

    best_gpr = grid.best_estimator_
    print(f"Best kernel: {best_gpr.kernel_}")

    y_train_pred = best_gpr.predict(X_train)
    y_test_pred  = best_gpr.predict(X_test)

    train_metrics = evaluate(y_train, y_train_pred, "TRAIN", y)
    test_metrics  = evaluate(y_test,  y_test_pred,  "TEST",  y)
    r2_train     = train_metrics[3]
    r2_test      = test_metrics[3]
    nrmse_train  = train_metrics[4]   # NRMSE_range
    nrmse_test   = test_metrics[4]

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_train, y_train_pred, alpha=0.6,
                label=f"Train  (R² = {r2_train:.4f}, NRMSE = {nrmse_train*100:.2f}%)",
                color=plt.cm.viridis(0.15))
    plt.scatter(y_test, y_test_pred, alpha=0.85,
                label=f"Test   (R² = {r2_test:.4f}, NRMSE = {nrmse_test*100:.2f}%)",
                color=plt.cm.viridis(0.85))
    plt.plot(lims, lims, "k--", lw=1, label="y = x")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"GPR — {sheet}")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTDIR, f"gpr_parity_{tag}.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

    return train_metrics, test_metrics


# -------------------------------------------------------------------
# Run on both sheets
# -------------------------------------------------------------------
results = {}
results["Beaker"]       = run_gpr("Beaker_Only",       BEAKER_FEATURES, "Au_extraction_%", "beaker",      [0, 1.2])
results["Microfluidic"] = run_gpr("Microfluidic_Only", MICRO_FEATURES,  "Au_extraction_%", "microfluidic", [0, 100])

print("\n" + "=" * 70)
print("  SUMMARY (GPR)")
print("=" * 70)
print(f"{'Sheet':<14}{'Set':<8}{'RMSE':>10}{'R²':>10}{'NRMSE_rng':>12}{'NRMSE_mean':>12}")
for name, (tr, te) in results.items():
    print(f"{name:<14}{'TRAIN':<8}{tr[0]:>10.4f}{tr[3]:>10.4f}{tr[4]:>12.4f}{tr[5]:>12.4f}")
    print(f"{name:<14}{'TEST':<8}{te[0]:>10.4f}{te[3]:>10.4f}{te[4]:>12.4f}{te[5]:>12.4f}")