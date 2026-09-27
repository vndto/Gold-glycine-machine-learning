# Analysis

This folder contains the interpretation and correlation analysis scripts for the gold–glycine leaching datasets.

## Datasets

The study uses two experimental datasets:

1. **Microfluidic** — gold leaching experiments carried out in a microfluidic setup
2. **Beaker** — conventional batch leaching experiments in a beaker

## Model-dependent analysis

Model-dependent analyses (feature importance, SHAP, and partial dependence plots) were performed on **one model per dataset**, rather than on all four models:

| Dataset     | Model used for analysis |
|-------------|-------------------------|
| Microfluidic | Random Forest (RF)     |
| Beaker      | XGBoost                 |

[Reason for selection, e.g. "Each model was chosen because it gave the lowest NRMSE on the test set for that dataset."]

The analyses included are:

1. **Feature importance** — ranks input variables by their contribution to model predictions
2. **SHapley Additive exPlanations (SHAP)** — quantifies each feature's contribution to individual predictions and its overall direction of effect
3. **Partial dependence plots (PDP)** — shows the marginal effect of one or two features on predicted gold extraction while averaging over the others

## Model-independent analysis

Pearson and Spearman correlation coefficients are calculated directly from the preprocessed data and do not depend on any model:

1. **Pearson** — linear correlation between each input feature and the target
2. **Spearman** — rank-based (monotonic) correlation, which is less sensitive to non-linearity and outliers

## Usage

Run each script from the repository root so the data file path resolves correctly, for example:

```bash
python "Analysis/[script name].py"
```

Results correspond to the interpretation figures in Chapter [X] of the thesis.
