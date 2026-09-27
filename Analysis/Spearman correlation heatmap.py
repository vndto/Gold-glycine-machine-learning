"""
Spearman correlation heatmap: process variables and targets
Data:     Beaker_Only AND Microfluidic_Only sheets (run separately)
Targets:  Au_extraction_%

Beaker and Microfluidic are never pooled — they have different feature
sets (several process variables are constant/zero in Microfluidic) and the
correlation structure of one system says nothing about the other's.

Spearman uses rank correlation rather than raw values, so it also picks up
monotonic-but-nonlinear relationships that Pearson (see the sibling
"Correlation Analysis" folder) would understate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(OUTDIR, "Preprocessed data.xlsx")

TARGETS = ["Au_extraction_%"]

BEAKER_FEATURES = [
    "Glycine_M", "Time_h", "Cyanide_ppm", "Cu_ppm", "KMnO4_M", "H2O2_%",
    "Temp_C", "pH",
]

MICRO_FEATURES = [
    "Glycine_M", "Time_h", "H2O2_%",
    "Temp_C", "pH",
]

DATASETS = {
    "beaker":       dict(sheet="Beaker_Only",       columns=BEAKER_FEATURES + TARGETS),
    "microfluidic": dict(sheet="Microfluidic_Only",  columns=MICRO_FEATURES + TARGETS),
}


# -------------------------------------------------------------------
# Heatmap for one dataset
# -------------------------------------------------------------------
def plot_corr_heatmap(tag, sheet, columns):
    print("\n" + "=" * 70)
    print(f"  Spearman correlation — {tag}  ({sheet})")
    print("=" * 70)

    df = pd.read_excel(FILE, sheet_name=sheet)

    # Spearman correlation is undefined (NaN) for a zero-variance column —
    # drop any before plotting instead of leaving blank cells in the map.
    nunique = df[columns].nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        print(f"Dropping constant columns (no variance in {sheet}): {constant}")
        columns = [c for c in columns if c not in constant]

    corr = df[columns].corr(method="spearman")

    print(corr.round(3).to_string())

    csv_name = os.path.join(OUTDIR, f"correlation_spearman_{tag}.csv")
    corr.to_csv(csv_name)
    print(f"Saved: {csv_name}")

    n = len(columns)
    fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 2.5))
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(corr.values, cmap=cmap, norm=norm)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)

    for i in range(n):
        for j in range(n):
            r, g, b, _ = cmap(norm(corr.values[i, j]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.85)
    ax.set_title(f"Spearman correlation — {tag}")
    fig.tight_layout()

    fname = os.path.join(OUTDIR, f"correlation_heatmap_spearman_{tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

    # Which features correlate most strongly with each target, at a glance.
    for target in TARGETS:
        if target not in corr.columns:
            continue
        ranked = (corr[target].drop(index=TARGETS, errors="ignore")
                  .abs().sort_values(ascending=False))
        print(f"\nTop correlations with {target}:")
        for feat, abs_rho in ranked.items():
            print(f"  {feat:<28} rho = {corr.loc[feat, target]:+.3f}")

    return corr


# -------------------------------------------------------------------
# Run on both sheets independently
# -------------------------------------------------------------------
results = {}
for tag, cfg in DATASETS.items():
    results[tag] = plot_corr_heatmap(tag, cfg["sheet"], cfg["columns"])
