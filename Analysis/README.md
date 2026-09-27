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

Each model was selected because it gave the best predictive performance on its dataset, with the highest coefficient of determination (R²) and the lowest normalized root mean square error (NRMSE) among the four models tested (RF, SVM, GPR, and XGBoost).

The analyses included are:

1. **Feature importance** — ranks input variables by their contribution to model predictions
2. **SHapley Additive exPlanations (SHAP)** — quantifies each feature's contribution to individual predictions and its overall direction of effect
3. **Partial dependence plots (PDP)** — shows the marginal effect of one or two features on predicted gold extraction while averaging over the others

## Model-independent analysis

Pearson and Spearman correlation coefficients are calculated directly from the preprocessed data and do not depend on any model:

1. **Pearson** — linear correlation between each input feature and the target
2. **Spearman** — rank-based (monotonic) correlation, which is less sensitive to non-linearity and outliers

