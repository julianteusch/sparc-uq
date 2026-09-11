# Method and Provenance

## Reference Inspected

The existing research checkout on bigbo is
`/home/julianteusch/dev/simlpe-pie-mp`, commit
`502289ab5e2e9779f331f41a7bd5732929bd5d2f`. It was clean when inspected on
2026-09-11. No files in that checkout were modified for this extraction.

Relevant reference modules:

- `src/simple_mp/hpo/trainers/hybrid_kappa.py`: global and horizon-wise feature
  sufficient statistics and inverse-DCT design construction.
- `src/simple_mp/hpo/evaluators/hybrid.py`: horizon-wise leverage, row scaling,
  optional pooled shrinkage, power transforms and CP-only joint scaling.
- `src/simple_mp/hpo/trainers/conjugate_bayes_fc_out.py`: full conjugate variants,
  which additionally update coefficients and are not the frozen-mean interface.

This repository is a fresh, dependency-light implementation of the reusable
equations, not a wholesale copy of the benchmark source or its Git history.
It intentionally avoids dataset paths, credentials, checkpoints and HPO machinery.

## Implemented vs. Not Yet Ported

| Component | Status |
| --- | --- |
| Frozen point mean | Implemented and tested |
| Per-horizon precision and raw kappa >= 1 | Implemented, float64 Cholesky solves |
| Shared-head inverse-DCT design including transformed bias | Helper, algebraic test |
| Scaling a supplied covariance factor | Implemented, correlation-preservation test |
| Graph-temporal construction | Small dense helper, specified factors only |
| Fitting a graph-temporal neural covariance head | Not ported |
| Residual covariance without a supplied head | Simple diagonal RMS fallback |
| Pooled kappa shrinkage and power stabilization | Not ported |
| CP-only joint kappa and pooled coordinate recipe | Not ported |
| Per-coordinate marginal split CP | Implemented; deliberately no coordinate pooling |
| Simultaneous max-score split CP | Generic extension, not paper default |
| Paper's exact datasets, checkpoints, metrics and HPO | Not ported |
| Arbitrary sklearn-like callable | Possible through explicit Forecast adapter; not automatically Bayesian |
| Efficient sparse/Kronecker production inference | Not yet implemented |

## Verification Levels

1. Unit identities: precision solve vs explicit inverse, analytic scalar Bayesian
   predictive variance, DCT reconstruction, SPD and correlation preservation.
2. API behavior: unchanged means, exactly one underlying model evaluation during
   prediction, round trips, calibration invalidation and rejected malformed inputs.
3. Synthetic integration: ordinary point regression and structured trajectory
   covariance. Reported numbers are not motion forecasting benchmark results.
4. Optional reference parity: actual pure numerical functions from the unchanged
   research checkout are compared with this implementation on deterministic tensors.
   Enable with `SPARC_REFERENCE_ROOT=/home/julianteusch/dev/simlpe-pie-mp`.
5. **Outstanding:** run one complete, locked paper configuration with its real
   preprocessing, pretrained mean and learned covariance before claiming full
   reproduction or checkpoint compatibility. Numerical core parity is not enough.

## Scientific Boundaries

Feature leverage describes support in a chosen representation. It is not a
general-purpose detector of distribution shift. A pretrained deterministic mean
need not coincide with the posterior mean of an exact conjugate Bayesian model.
These qualifications do not invalidate split conformal calibration if its score
function is fixed and the calibration/test units are exchangeable.

For fixed-coordinate marginal scores, dependence between coordinates within one
row is allowed; exchangeability is needed across the selected rows. Randomly
splitting highly overlapping time windows does not establish this assumption.

References:
- [SPARC](https://arxiv.org/abs/2608.20802).
- [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).
- [PyTorch triangular solves](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html).
- [PyTorch serialization API](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
