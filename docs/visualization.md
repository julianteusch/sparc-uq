# Visualizations and Design-Space Explanations

## Notebook

Open [`notebooks/01_forecasts_and_diagnostics.ipynb`](../notebooks/01_forecasts_and_diagnostics.ipynb).
It includes executed outputs for a fixed, synthetic two-joint forecasting example.
The graph-temporal covariance is specified, not learned; this is not a paper
benchmark reproduction or a demonstration of superiority over a baseline.

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/01_forecasts_and_diagnostics.ipynb
```

On a remote machine, run Jupyter bound to localhost and access it through an SSH
tunnel. Do not expose an unauthenticated Jupyter server publicly. No server is
started automatically by this package or its notebook.

The paired percent-format Python file is the editable source. To regenerate and
execute the notebook without a browser, from the repository root:

```bash
python scripts/execute_notebook.py
```

This uses the invoking Python environment as the kernel, fails on cell errors,
checks that six plot outputs were produced and saves an executed `.ipynb`. The
notebook exports PNG/SVG figures and a synthetic UQ checkpoint into ignored
`artifacts/notebook/`. CI executes the notebook on Python 3.12; the paired `.py`
source is linted instead of rewriting output-bearing notebook JSON.

## Reusable Plotting API

Install only `.[visualization]` if you do not need Jupyter. Core imports remain
PyTorch-only. Import plotting explicitly:

```python
from sparc.visualization import plot_forecast, plot_calibration

prediction = uq.predict(test_inputs)
ax = plot_forecast(prediction, test_targets, sample=0, channel=0)
ax.figure.savefig("forecast.svg", bbox_inches="tight")
axes = plot_calibration(prediction, test_targets, nominal=1 - uq.calibration.alpha)
```

| Helper | Input / Interpretation |
| --- | --- |
| `plot_forecast` | One coordinate's point mean, discrete conformal intervals and optional future targets |
| `plot_calibration` | Mean marginal coverage and mean width per horizon on supplied held-out rows |
| `plot_covariance` | Square covariance, optionally converted to correlation; use shared limits for comparisons |
| `plot_feature_support` | A two-dimensional projection of the design rows, colored by supplied kappa |
| `plot_leverage_terms` | Exact signed decomposition of the leverage quadratic form |
| `plot_risk_diagnostic` | Mean-horizon kappa versus observed per-row RMSE; descriptive association only |

Helpers accept Matplotlib axes, return them, and never call `show`, select a
backend or modify global plotting settings. They detach/copy tensors to CPU for
plotting without changing fitted state. Use `MPLBACKEND=Agg` for headless scripts.
No API computes SHAP values, counterfactuals or causal attributions.

Infinite intervals are rejected with an explicit error, not silently clipped.
`plot_calibration` always displays mean marginal coverage, even when the caller
used a simultaneous calibration scope; it is not a whole-row coverage statistic.
The caller supplies the appropriate nominal level. It draws no binomial error
bars because coordinates within one row may be dependent.

## Exact Leverage Diagnostics

```python
from sparc import explain_leverage
from sparc.visualization import plot_leverage_terms

features = adapter(test_inputs).features  # [B,H,P], including the correct bias
explanation = explain_leverage(uq.leverage, features)
ax = plot_leverage_terms(explanation, sample=0, horizon=0, feature_names=design_names, top_k=6)
```

For fixed fitted precision Lambda:

```text
c_p = phi_p * (Lambda^-1 phi)_p
sum_p c_p = kappa - 1
gradient_phi(kappa) = 2 * Lambda^-1 phi
```

`LeverageExplanation.terms` and `.sensitivity` have shape `[B,H,P]` and remain on
the query tensor's device; `.kappa` has shape `[B,H]`. Computations use float64
and Cholesky solves. No new fit, stochastic sampling or model mutation occurs.
This is additional diagnostic computation, not part of a zero-cost attribution
claim or the one-forward inference guarantee.

Important boundaries:

- Terms can be negative because features interact through the precision matrix.
  A negative term does not mean that deleting that feature would reduce error.
- Terms depend on the chosen design/basis and prior. Sensitivities depend on
  coordinate units and hold the fitted precision fixed. They are gradients with
  respect to **design features**, not raw inputs or model error.
- A transformed bias is a model coordinate, not a measured causal input.
- With `top_k`, omitted terms are aggregated so the displayed bars still sum to
  kappa-1. The aggregate may cancel internally; inspect all terms when needed.
- A two-dimensional projection omits other design dimensions and does not define
  the full Mahalanobis geometry.
- Large kappa does not automatically imply a bad mean forecast, OOD detection,
  conditional coverage or safe deployment. Check associations on appropriate
  validation data before defining a risk policy; do not tune on the final test set.

See the [SPARC paper](https://arxiv.org/abs/2608.20802) for the method and empirical
risk diagnostic. The additive decomposition and design-gradient visualizations
here are algebraic inspection tools, not a newly validated explanation method.
