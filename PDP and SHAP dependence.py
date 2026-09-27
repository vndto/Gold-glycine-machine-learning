"""
Partial dependence (PDP) and SHAP dependence: Random Forest vs XGBoost
Data:     Beaker_Only AND Microfluidic_Only sheets (run separately)
Target:   Au_extraction_%

Same modeling pipeline as "SHAP Analysis/SHAP analysis.py" (same tuning
grids, same plain random train_test_split), so the models explained here
match the ones whose SHAP importance is reported there. This folder is
self-contained (its own copy of the data, its own fit) and adds the
per-feature shape of each model's response:
  - PDP:  2-way (pairwise) partial dependence between every pair among
          each run's own top-3 SHAP-ranked features — the model's average
          predicted response over a (feature_x, feature_y) grid, holding
          every other feature at its observed distribution — drawn as a
          3D surface, with the grid point of maximum partial dependence
          (the pairwise optimum for that model) marked and tabulated in
          pdp_optima_<model>_<dataset>.csv. Both models are piecewise
          constant, so that maximum is usually a flat plateau rather than
          a point; the tied grid points are marked too and their extent
          is reported (n_tied, {x,y}_plateau_{min,max}), because quoting
          one corner of a plateau as "the optimum" would read as a
          precision the model does not have. Note that a PDP optimum is an
          average over the marginal distribution of the features NOT on
          the axes, so its height is well below the best single observed
          run whenever the best runs need a specific JOINT setting of
          more features than the two being swept (which is the case for
          Beaker: its >1.2 extraction rows also need Glycine_M = 0.05 and
          no H2O2/Cu/cyanide). Read the marked point as "best setting of
          this pair, averaging over everything else", not as an
          achievable yield. "Top three"
          is determined separately per run (mean |SHAP value| on that
          run's own test explanation), never a shared/global ranking,
          consistent with never averaging SHAP across models or datasets.
          Computed manually (classic Friedman PDP, generalized to 2
          features) rather than via sklearn's PartialDependenceDisplay —
          this xgboost version predates sklearn's estimator-tags protocol,
          which makes sklearn's is_regressor() check, and so its built-in
          PDP tools, fail for XGBRegressor specifically; a hand-rolled
          version works identically for both models regardless of that
          mismatch.
  - SHAP dependence: each test-set row's SHAP value for one feature,
          plotted against that feature's value (colored by the feature it
          interacts with most) — shows the same per-feature effect at the
          individual-prediction level, plus interaction structure PDP
          averages away.

Beaker and Microfluidic are never pooled — different feature sets, and
the target lives on different scales (Beaker ~0-1.3, Microfluidic ~0-100).
"""

import math
import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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

# Grid the pair over the FULL observed range of each feature (0th-100th
# percentile) rather than the usual 5th-95th trim: the optimum of several
# beaker features sits in the tail the trim would cut off (e.g. the best
# observed Time_h is 76 h against a 95th percentile of 71 h), and clipping
# it would move the reported pairwise optimum.
PDP_PERCENTILES = (0.0, 1.0)
GRID_RESOLUTION = 30

# 3D surface viewing angle (elevation, azimuth) in degrees.
VIEW_INIT = (28, -135)

DATASETS = {
    "beaker":       dict(sheet="Beaker_Only",      features=BEAKER_FEATURES),
    "microfluidic": dict(sheet="Microfluidic_Only", features=MICRO_FEATURES),
}

# Same tuning grids as RF.py / XGBoost.py / SHAP analysis.py, so the "best"
# model explained here is the same model those scripts report metrics for.
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

# Every pairwise optimum found, appended per dataset/model, written out as
# one combined table at the end.
ALL_OPTIMA = []


# -------------------------------------------------------------------
# Manual partial dependence (see module docstring for why not sklearn's)
# -------------------------------------------------------------------
def compute_pdp_2d(model, X_ref, feat_x, feat_y, grid_resolution=GRID_RESOLUTION,
                    percentiles=PDP_PERCENTILES):
    """Classic Friedman PDP generalized to a pair of features: sweep both
    across their PDP_PERCENTILES range on a grid_resolution x
    grid_resolution grid, holding every other row's other features fixed
    at their observed values, and average the model's predictions at each
    grid point.
    """
    x_lo, x_hi = X_ref[feat_x].quantile(percentiles[0]), X_ref[feat_x].quantile(percentiles[1])
    y_lo, y_hi = X_ref[feat_y].quantile(percentiles[0]), X_ref[feat_y].quantile(percentiles[1])
    x_grid = np.linspace(x_lo, x_hi, grid_resolution)
    y_grid = np.linspace(y_lo, y_hi, grid_resolution)
    Z = np.empty((grid_resolution, grid_resolution))
    X_mod = X_ref.copy()
    for i, yv in enumerate(y_grid):
        X_mod[feat_y] = yv
        for j, xv in enumerate(x_grid):
            X_mod[feat_x] = xv
            Z[i, j] = model.predict(X_mod).mean()
    return x_grid, y_grid, Z


# -------------------------------------------------------------------
# Grid-of-subplots helper
# -------------------------------------------------------------------
def make_grid(n, n_cols=4, panel_w=4.2, panel_h=3.4, projection=None):
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(panel_w * n_cols, panel_h * n_rows),
                              squeeze=False,
                              subplot_kw={"projection": projection} if projection else None)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat[:n]


# -------------------------------------------------------------------
# PDP + SHAP dependence for one dataset
# -------------------------------------------------------------------
def run_analysis(tag, sheet, features):
    print("\n" + "=" * 70)
    print(f"  PDP / SHAP dependence — {tag}  ({sheet})")
    print("=" * 70)

    df = pd.read_excel(FILE, sheet_name=sheet)

    # A zero-variance feature can't be split on or partial-dependence'd —
    # drop it before fitting instead of failing partway through.
    nunique = df[features].nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        print(f"[{tag}] dropping constant feature(s): {constant}")
        features = [f for f in features if f not in constant]

    # Integer-dtype columns (e.g. Cyanide_ppm, Temp_C) trigger a sklearn
    # FutureWarning in partial_dependence and will error in a future
    # release — cast once, up front.
    X = df[features].astype(float)
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    for model_tag, fitter in MODEL_FITTERS.items():
        print(f"\n--- Tuning {model_tag.upper()} on {tag} ---")
        model = fitter(X_train, y_train)

        # ---- SHAP explanation (background AND explanation set both this --
        # ---- dataset's own held-out test split — never shared across ----
        # ---- datasets); needed both for SHAP dependence below and to ----
        # ---- rank this run's own top-3 features for pairwise PDP.
        explainer = shap.Explainer(model, X_test)
        explanation = explainer(X_test)

        mean_abs_shap = pd.Series(np.abs(explanation.values).mean(axis=0),
                                   index=features).sort_values(ascending=False)
        top3 = mean_abs_shap.index[:3].tolist()
        print(f"[{tag}/{model_tag}] top-3 by mean |SHAP|: "
              f"{', '.join(f'{f} ({mean_abs_shap[f]:.4f})' for f in top3)}")

        # ---- Pairwise PDP among this run's own top-3 SHAP features ------
        # (PDP marginalizes over the feature distribution; train reflects
        # the distribution the model was actually fit on.)
        pairs = list(combinations(top3, 2))
        fig, axes = make_grid(len(pairs), n_cols=3, panel_w=5.8, panel_h=4.8,
                              projection="3d")
        pdp_rows = []
        opt_rows = []
        for ax, (feat_x, feat_y) in zip(axes, pairs):
            x_grid, y_grid, Z = compute_pdp_2d(model, X_train, feat_x, feat_y)
            XX, YY = np.meshgrid(x_grid, y_grid)
            surf = ax.plot_surface(XX, YY, Z, cmap="viridis", linewidth=0,
                                   antialiased=True, alpha=0.92,
                                   rstride=1, cstride=1)

            # Pairwise optimum: the grid point with the highest partial
            # dependence, i.e. the best (feature_x, feature_y) setting once
            # every other feature is averaged over its observed distribution.
            #
            # Both models are tree ensembles, so the surface is piecewise
            # constant and the maximum is normally a PLATEAU, not a point:
            # a feature with three observed levels (KMnO4_M) gives the same
            # prediction everywhere above its last split threshold. A bare
            # argmax would then report one arbitrary corner of that plateau
            # as "the optimum" — so find every tied grid point, report the
            # plateau's extent, and mark the tied point nearest the
            # plateau's centroid as its representative.
            tied = np.isclose(Z, Z.max(), rtol=1e-9, atol=0.0)
            iy_t, ix_t = np.nonzero(tied)
            n_tied = int(iy_t.size)
            # Centroid in grid-index space, so the pick is unaffected by the
            # very different units of the two axes.
            k = np.argmin((iy_t - iy_t.mean()) ** 2 + (ix_t - ix_t.mean()) ** 2)
            iy, ix = int(iy_t[k]), int(ix_t[k])
            x_opt, y_opt, z_opt = x_grid[ix], y_grid[iy], Z[iy, ix]
            x_plateau = (x_grid[ix_t].min(), x_grid[ix_t].max())
            y_plateau = (y_grid[iy_t].min(), y_grid[iy_t].max())
            if n_tied > 1:
                ax.scatter(x_grid[ix_t], y_grid[iy_t], Z[iy_t, ix_t],
                           color="crimson", s=7, alpha=0.55, depthshade=False,
                           zorder=9)
            ax.scatter([x_opt], [y_opt], [z_opt], color="crimson", s=45,
                       depthshade=False, zorder=10)
            # A dropline makes the optimum's position readable on the axes,
            # which a bare 3D marker never is.
            ax.plot([x_opt, x_opt], [y_opt, y_opt], [Z.min(), z_opt],
                    color="crimson", lw=1.0, ls="--")
            ax.text(x_opt, y_opt, z_opt, f"  {z_opt:.3f}", color="crimson",
                    fontsize=8, zorder=11)

            ax.set_xlabel(feat_x, fontsize=10, fontweight="bold", labelpad=4)
            ax.set_ylabel(feat_y, fontsize=10, fontweight="bold", labelpad=4)
            ax.set_zlabel("Partial dependence", fontsize=9, labelpad=4)
            ax.tick_params(labelsize=7)
            plateau_note = "" if n_tied == 1 else (
                f"\nplateau ({n_tied} pts): {feat_x} {x_plateau[0]:.3g}-{x_plateau[1]:.3g}, "
                f"{feat_y} {y_plateau[0]:.3g}-{y_plateau[1]:.3g}")
            ax.set_title(f"opt: {feat_x}={x_opt:.4g}, {feat_y}={y_opt:.4g}\n"
                         f"PD = {z_opt:.4f}{plateau_note}", fontsize=9, pad=2)
            ax.view_init(elev=VIEW_INIT[0], azim=VIEW_INIT[1])
            # pad kept small: on a 3D axes the colorbar pad is measured from the
            # axes box, and a large one lands on the next panel's z ticks.
            cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.02, aspect=16)
            cb.ax.tick_params(labelsize=7)

            opt_rows.append({
                "dataset": tag, "model": model_tag,
                "feature_x": feat_x, "feature_y": feat_y,
                "x_opt": x_opt, "y_opt": y_opt, "pdp_max": z_opt,
                "n_tied": n_tied, "n_grid_points": Z.size,
                "x_plateau_min": x_plateau[0], "x_plateau_max": x_plateau[1],
                "y_plateau_min": y_plateau[0], "y_plateau_max": y_plateau[1],
                "pdp_min": Z.min(),
                "x_min_swept": x_grid[0], "x_max_swept": x_grid[-1],
                "y_min_swept": y_grid[0], "y_max_swept": y_grid[-1],
            })
            for yi, yv in enumerate(y_grid):
                for xi, xv in enumerate(x_grid):
                    pdp_rows.append({
                        "feature_x": feat_x, "feature_y": feat_y,
                        "x_value": xv, "y_value": yv, "pdp": Z[yi, xi],
                    })
        fig.suptitle(f"Pairwise partial dependence (top-3 SHAP features) — "
                     f"{model_tag.upper()} — {tag}", fontsize=13, y=1.03)
        # rect/pad_inches: matplotlib's tight bbox misses a 3D axes' z-axis
        # label, so the leftmost panel loses it without a little slack.
        fig.tight_layout(w_pad=2.5, rect=[0.012, 0, 1, 1])
        fname = os.path.join(OUTDIR, f"pdp_pairwise_3d_{model_tag}_{tag}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        print(f"Saved: {fname}")

        pdp_csv = os.path.join(OUTDIR, f"pdp_pairwise_values_{model_tag}_{tag}.csv")
        pd.DataFrame(pdp_rows).to_csv(pdp_csv, index=False)
        print(f"Saved: {pdp_csv}")

        opt_df = pd.DataFrame(opt_rows)
        opt_csv = os.path.join(OUTDIR, f"pdp_optima_{model_tag}_{tag}.csv")
        opt_df.to_csv(opt_csv, index=False)
        print(f"Saved: {opt_csv}")
        print(f"[{tag}/{model_tag}] pairwise optima:")
        print(opt_df[["feature_x", "feature_y", "x_opt", "y_opt",
                      "pdp_max", "n_tied"]].to_string(index=False))
        ALL_OPTIMA.append(opt_df)

        # ---- SHAP dependence: per-row SHAP value vs. feature value ------
        fig, axes = make_grid(len(features))
        for ax, feat in zip(axes, features):
            shap.plots.scatter(explanation[:, feat], ax=ax, show=False)
            ax.set_xlabel(feat, fontsize=10, fontweight="bold")
        fig.suptitle(f"SHAP dependence — {model_tag.upper()} — {tag}",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fname = os.path.join(OUTDIR, f"shap_dependence_{model_tag}_{tag}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

        dep_df = pd.DataFrame(explanation.values, columns=features)
        dep_df = dep_df.add_suffix("_shap")
        for feat in features:
            dep_df[feat] = X_test[feat].to_numpy()
        dep_csv = os.path.join(OUTDIR, f"shap_dependence_values_{model_tag}_{tag}.csv")
        dep_df.to_csv(dep_csv, index=False)
        print(f"Saved: {dep_csv}")


# -------------------------------------------------------------------
# Run each dataset independently
# -------------------------------------------------------------------
for tag, cfg in DATASETS.items():
    run_analysis(tag, cfg["sheet"], cfg["features"])

# One combined table of every pairwise optimum (dataset x model x pair),
# kept separate from the per-run files so nothing here implies pooling:
# the rows are simply concatenated, never averaged across models/datasets.
if ALL_OPTIMA:
    summary = pd.concat(ALL_OPTIMA, ignore_index=True)
    summary_csv = os.path.join(OUTDIR, "pdp_optima_all.csv")
    summary.to_csv(summary_csv, index=False)
    print("\n" + "=" * 70)
    print("  Pairwise PDP optima — all datasets / models")
    print("=" * 70)
    print(summary[["dataset", "model", "feature_x", "feature_y",
                   "x_opt", "y_opt", "pdp_max", "n_tied"]].to_string(index=False))
    print(f"\nSaved: {summary_csv}")
