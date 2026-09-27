"""
Machine learning study: Gold leaching with glycine
Model:    Random Forest with hyperparameter tuning
Data:     Beaker_Only AND Microfluidic_Only sheets (run separately)
Target:   Au_extraction_%
Split:    80% train / 20% test  (optionally group-aware by series)

Metrics: RMSE, MAE, MAPE, R2, NRMSE_range, NRMSE_mean

CHANGES vs. the original script
-------------------------------
1. Pre-fit DIAGNOSTICS: constant features, NaN counts, and — the key one —
   duplicate feature vectors whose target is not constant. These are the rows
   that produce the horizontal band at y_pred = mean(y_train): the trees cannot
   split them, so every one collapses onto the same leaf mean.
2. Zero-variance features are dropped automatically (they cost splits and,
   per your leave-series-out work, several of them are study fingerprints).
3. CURATION is physical/structural, never residual-based:
     - out-of-range targets (<0 or >100) removed
     - within-series backsliding beyond digitisation tolerance flagged
     - unidentifiable duplicate groups either reported, collapsed to a
       weighted mean, or dropped — chosen by a flag, not by error size.
4. Train predictions are reported OOB as well as in-sample, so the train/test
   R2 comparison is apples-to-apples.
5. The parity plot marks the stuck-at-mean rows so the band is visible as a
   diagnosis rather than as noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, GridSearchCV, KFold, GroupKFold,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FILE = "Preprocessed_data_normalized.xlsx"

BEAKER_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH", "Surface_area_cm2", "Agitation_rpm",
    "Solid_to_volume_ratio_g/L",
]

MICRO_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH", "Surface_area_cm2",
    "Solid_to_volume_ratio_g/L",
]

PARAM_GRID = {
    "n_estimators":      [100, 200, 500],
    "max_depth":         [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 1.0],
}

# --- curation switches -------------------------------------------------
DROP_CONSTANT_FEATURES = True     # remove zero-variance columns
TARGET_MIN, TARGET_MAX = -1.0, 101.0   # physical bounds for % extraction
MONOTONE_TOL_PP        = 2.0      # allowed backslide (percentage points)
DROP_BACKSLIDERS       = False    # True once you trust MONOTONE_TOL_PP
DUPLICATE_POLICY       = "report" # "report" | "collapse" | "drop"
DUP_ROUND              = 6        # decimals used to define an identical X
GROUP_AWARE_SPLIT      = False    # True -> leave-series-out style split + CV
SERIES_CANDIDATES      = ["Series_ID", "series_id", "Series", "Study_ID",
                          "Study", "Ref", "Reference", "Source"]


# ===================================================================
# Diagnostics
# ===================================================================
def find_series_column(df):
    for c in SERIES_CANDIDATES:
        if c in df.columns:
            return c
    return None


def diagnose(df, features, target, sheet):
    """Report everything that could make rows unfittable, before any modelling."""
    print("\n" + "-" * 60)
    print(f"  DIAGNOSTICS — {sheet}")
    print("-" * 60)

    X = df[features]
    y = df[target]

    # --- 1. constant / near-constant features -------------------------
    nunique = X.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    two_level = nunique[nunique == 2].index.tolist()
    print(f"\n[1] Unique values per feature (n={len(df)} rows):")
    for c in features:
        if c in constant:
            flag = "  <-- CONSTANT, no split possible"
        elif c in two_level:
            flag = "  <-- two-level: direction only, not a reliable importance rank"
        else:
            flag = ""
        print(f"      {c:<28} {nunique[c]:>5}{flag}")

    # --- 2. missing values --------------------------------------------
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print("\n[2] Missing values:")
        for c in features:
            if nan_counts[c]:
                print(f"      {c:<28} {nan_counts[c]:>5} NaN "
                      f"({100*nan_counts[c]/len(df):.1f}%)")
    else:
        print("\n[2] Missing values: none")

    # --- 3. duplicate feature vectors with non-constant target --------
    usable = [c for c in features if c not in constant]
    key = X[usable].round(DUP_ROUND).astype(object).fillna("__NaN__")
    gid = pd.factorize(pd.Series(
        [tuple(r) for r in key.itertuples(index=False)], dtype=object))[0]
    stats = (pd.DataFrame({"y": y.values, "gid": gid})
             .groupby("gid")["y"].agg(["count", "min", "max", "std"]))
    stats["spread"] = stats["max"] - stats["min"]
    bad = stats[(stats["count"] > 1) & (stats["spread"] > MONOTONE_TOL_PP)]

    n_bad_rows = int(bad["count"].sum())
    print(f"\n[3] Identical feature vectors with differing target:")
    print(f"      groups : {len(bad)}")
    print(f"      rows   : {n_bad_rows}  ({100*n_bad_rows/len(df):.1f}% of data)")
    if len(bad):
        print(f"      worst spreads (target range within one identical X):")
        for k, r in bad.sort_values('spread', ascending=False).head(10).iterrows():
            print(f"        n={int(r['count']):>3}  "
                  f"y in [{r['min']:.2f}, {r['max']:.2f}]  "
                  f"spread={r['spread']:.2f}")
        print("\n      -> These rows CANNOT be fitted by any model on this feature")
        print("         set. A Random Forest returns their group mean, which is")
        print("         what produces the horizontal band on the parity plot.")
        print("         Fix upstream (recover the missing discriminator, e.g.")
        print("         Time_h for that series) rather than deleting residuals.")

    # --- 4. what the band value will be -------------------------------
    print(f"\n[4] mean({target}) = {y.mean():.4f}  "
          f"<- expected y-position of the flat band")

    return constant, bad


def unidentifiable_mask(df, features, target):
    """True for rows sharing an identical feature vector with a different target."""
    usable = [c for c in features if df[c].nunique(dropna=False) > 1]
    if not usable:
        return np.ones(len(df), dtype=bool)
    key = df[usable].round(DUP_ROUND).astype(object).fillna("__NaN__")
    gid = pd.factorize(pd.Series(
        [tuple(r) for r in key.itertuples(index=False)], dtype=object))[0]
    tmp = pd.DataFrame({"gid": gid, "y": df[target].values})
    g = tmp.groupby("gid")["y"].agg(["count", "min", "max"])
    g["spread"] = g["max"] - g["min"]
    bad = set(g.index[(g["count"] > 1) & (g["spread"] > MONOTONE_TOL_PP)])
    return np.isin(gid, list(bad))


# ===================================================================
# Curation  (structural / physical only — never residual-based)
# ===================================================================
def curate(df, features, target, series_col):
    n0 = len(df)
    log = []

    # -- physical bounds -----------------------------------------------
    bad_range = (df[target] < TARGET_MIN) | (df[target] > TARGET_MAX)
    if bad_range.any():
        log.append(f"{bad_range.sum()} rows outside [{TARGET_MIN}, {TARGET_MAX}] %")
        df = df.loc[~bad_range].copy()

    # -- within-series monotonicity ------------------------------------
    if series_col and "Time_h" in df.columns:
        df = df.sort_values([series_col, "Time_h"])
        back = df.groupby(series_col)[target].diff() < -MONOTONE_TOL_PP
        n_back = int(back.sum())
        if n_back:
            log.append(f"{n_back} rows backslide > {MONOTONE_TOL_PP} pp within a series"
                       + (" (dropped)" if DROP_BACKSLIDERS else " (flagged only)"))
            if DROP_BACKSLIDERS:
                df = df.loc[~back].copy()

    # -- unidentifiable duplicate groups -------------------------------
    usable = [c for c in features if df[c].nunique(dropna=False) > 1]
    key = df[usable].round(DUP_ROUND).astype(object).fillna("__NaN__")
    gid = pd.factorize(pd.Series(
        [tuple(r) for r in key.itertuples(index=False)], dtype=object))[0]
    df = df.assign(_gid=gid)
    gs = df.groupby("_gid")[target].agg(["count", "min", "max"])
    gs["spread"] = gs["max"] - gs["min"]
    bad_gids = gs.index[(gs["count"] > 1) & (gs["spread"] > MONOTONE_TOL_PP)]

    weights = None
    if len(bad_gids):
        if DUPLICATE_POLICY == "drop":
            keep = ~df["_gid"].isin(bad_gids)
            log.append(f"{(~keep).sum()} unidentifiable rows dropped "
                       f"({len(bad_gids)} groups)")
            df = df.loc[keep].copy()
        elif DUPLICATE_POLICY == "collapse":
            agg = {c: "first" for c in df.columns if c not in (target, "_gid")}
            agg[target] = "mean"
            n_before = len(df)
            counts = df.groupby("_gid").size()
            df = df.groupby("_gid", as_index=False).agg(agg)
            weights = counts.reindex(df["_gid"]).values.astype(float)
            log.append(f"{n_before - len(df)} rows collapsed into group means "
                       f"(sample_weight = group size)")
        else:
            log.append(f"{int(gs.loc[bad_gids,'count'].sum())} unidentifiable rows "
                       f"RETAINED (DUPLICATE_POLICY='report')")

    df = df.drop(columns="_gid").reset_index(drop=True)
    print(f"\n[curation] {n0} -> {len(df)} rows")
    for line in log:
        print(f"           - {line}")
    return df, weights


# -------------------------------------------------------------------
# Evaluation helper  (unchanged interface)
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


# ===================================================================
# Main routine
# ===================================================================
def run_rf(sheet, features, target, tag):
    print("\n" + "=" * 60)
    print(f"  Random Forest — {sheet}  (target: {target})")
    print("=" * 60)

    df = pd.read_excel(FILE, sheet_name=sheet)
    print(f"Loaded {sheet}: {df.shape[0]} rows")

    features = [f for f in features if f in df.columns]
    series_col = find_series_column(df)
    print(f"Series column detected: {series_col or 'NONE (group-aware CV unavailable)'}")

    # ---- 1. diagnose BEFORE touching anything -----------------------
    constant, bad_groups = diagnose(df, features, target, sheet)

    # ---- 2. drop zero-variance features -----------------------------
    if DROP_CONSTANT_FEATURES and constant:
        print(f"\n[features] dropping constant columns: {constant}")
        features = [f for f in features if f not in constant]
    print(f"[features] using {len(features)}: {features}")

    # ---- 3. curate ---------------------------------------------------
    df, weights = curate(df, features, target, series_col)

    X = df[features].values
    y = df[target].values
    groups = df[series_col].values if series_col else None
    print(f"\nTarget: {target} | range [{y.min():.4f}, {y.max():.4f}] | mean {y.mean():.4f}")

    # ---- 4. split ----------------------------------------------------
    idx = np.arange(len(y))
    if GROUP_AWARE_SPLIT and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(splitter.split(X, y, groups))
        print("Split: GROUP-AWARE (no series appears in both train and test)")
    else:
        tr, te = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
        print("Split: random 80/20  (NOTE: leaks within-series information)")

    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]
    w_train = weights[tr] if weights is not None else None
    print(f"Train: {len(tr)} | Test: {len(te)}")

    # ---- 5. tune -----------------------------------------------------
    if GROUP_AWARE_SPLIT and groups is not None:
        n_groups = len(np.unique(groups[tr]))
        cv = GroupKFold(n_splits=min(10, n_groups))
        fit_groups = groups[tr]
    else:
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        fit_groups = None

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True,
                              bootstrap=True),
        param_grid=PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    print("\nRunning grid search ...")
    fit_kw = {}
    if w_train is not None:
        fit_kw["sample_weight"] = w_train
    grid.fit(X_train, y_train, groups=fit_groups, **fit_kw)

    print("\n=== Best hyperparameters ===")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    best_rf = grid.best_estimator_
    y_train_pred = best_rf.predict(X_train)
    y_test_pred  = best_rf.predict(X_test)

    train_metrics = evaluate(y_train, y_train_pred, "TRAIN (in-sample)", y)
    test_metrics  = evaluate(y_test,  y_test_pred,  "TEST",  y)

    # ---- 6. OOB train score: comparable with TEST --------------------
    oob = getattr(best_rf, "oob_prediction_", None)
    if oob is not None and np.isfinite(oob).all():
        oob_metrics = evaluate(y_train, oob, "TRAIN (out-of-bag)", y)
        print(f"\n  -> Compare TEST R2 ({test_metrics[3]:.4f}) with the OOB R2 "
              f"({oob_metrics[3]:.4f}), not with the in-sample R2.")

    # ---- 7. which rows are structurally unfittable? -------------------
    unident = unidentifiable_mask(df, features, target)
    band, band_te = unident[tr], unident[te]
    print(f"\n[band] unidentifiable rows — TRAIN: {band.sum()} "
          f"({100*band.sum()/len(y_train):.1f}%) | TEST: {band_te.sum()} "
          f"({100*band_te.sum()/len(y_test):.1f}%)")
    if band.sum() and band.sum() / len(y_train) > 1.5 * max(
            band_te.sum() / max(len(y_test), 1), 1e-9):
        print("       -> these unfittable rows are concentrated in TRAIN, which")
        print("          depresses train R2 relative to test R2.")
    if band.sum():
        print(f"       -> RMSE on identifiable train rows only: "
              f"{np.sqrt(mean_squared_error(y_train[~band], y_train_pred[~band])):.4f} "
              f"(vs {train_metrics[0]:.4f} overall)")

    r2_train    = train_metrics[3]
    r2_test     = test_metrics[3]
    nrmse_train = train_metrics[4]
    nrmse_test  = test_metrics[4]

    # ---- 8. parity plot with the band marked -------------------------
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolor="k", color="steelblue",
                label=f"Train  (R² = {r2_train:.4f}, NRMSE = {nrmse_train*100:.2f}%)")
    plt.scatter(y_test, y_test_pred, alpha=0.85, edgecolor="k", color="orange",
                label=f"Test   (R² = {r2_test:.4f}, NRMSE = {nrmse_test*100:.2f}%)")
    if band.any():
        plt.scatter(y_train[band], y_train_pred[band], facecolors="none",
                    edgecolor="crimson", s=110, lw=1.4,
                    label=f"Unidentifiable (n = {band.sum()})")
        plt.axhline(y_train.mean(), color="crimson", ls=":", lw=1, alpha=0.6)
    lims = [min(y.min(), y_train_pred.min(), y_test_pred.min()),
            max(y.max(), y_train_pred.max(), y_test_pred.max())]
    plt.plot(lims, lims, "k--", lw=1, label="y = x")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Random Forest — {sheet}")
    plt.legend(loc="upper left", framealpha=0.9, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"rf_parity_{tag}.png", dpi=150)
    plt.show()

    return train_metrics, test_metrics


# -------------------------------------------------------------------
# Run on both sheets
# -------------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    results["Beaker"]       = run_rf("Beaker_Only", BEAKER_FEATURES,
                                     "Au_extraction_%", "beaker")
    results["Microfluidic"] = run_rf("Microfluidic_Only", MICRO_FEATURES,
                                     "Au_extraction_%", "microfluidic")

    print("\n" + "=" * 70)
    print("  SUMMARY (Random Forest)")
    print("=" * 70)
    print(f"{'Sheet':<14}{'Set':<8}{'RMSE':>10}{'R²':>10}{'NRMSE_rng':>12}{'NRMSE_mean':>12}")
    for name, (tr_m, te_m) in results.items():
        print(f"{name:<14}{'TRAIN':<8}{tr_m[0]:>10.4f}{tr_m[3]:>10.4f}"
              f"{tr_m[4]:>12.4f}{tr_m[5]:>12.4f}")
        print(f"{name:<14}{'TEST':<8}{te_m[0]:>10.4f}{te_m[3]:>10.4f}"
              f"{te_m[4]:>12.4f}{te_m[5]:>12.4f}")