# Changelog

## 0.1.0 - 2026-09-11

Initial alpha release of the reusable SPARC-UQ integration library, licensed under
Apache 2.0. Distribution: `sparc-uq`; import: `sparc`.

### Core

- Keep a frozen point estimator through a linear-last-layer adapter or an explicit
  callable returning forecasts and design features.
- Fit feature precision, compute horizon-wise leverage scales, and scale supplied
  diagonal or dense structured covariance factors without changing point means.
- Calibrate on held-out rows with marginal or simultaneous split conformal scores.
- Save and restore fitted uncertainty state separately from predictor weights.

### Notebook and Diagnostics

- Executed Jupyter notebook with six visualizations of a synthetic motion example.
- Optional Matplotlib helpers for forecasts, interval coverage and width,
  covariance/correlation, feature support, and kappa-versus-error diagnostics.
- Signed decomposition of kappa-1 and local design-feature sensitivities.
  These are descriptive diagnostics, not SHAP values or causal explanations.

### Scope

This is not the full paper benchmark implementation: no learned covariance-head
training, paper checkpoints, datasets, or reproduced benchmark tables are shipped.
Coverage relies on exchangeability and the selected prediction unit; it is not
conditional coverage or protection against arbitrary distribution shift.

Paper: https://arxiv.org/abs/2608.20802
Notebook: https://github.com/julianteusch/sparc-uq/blob/v0.1.0/notebooks/01_forecasts_and_diagnostics.ipynb
