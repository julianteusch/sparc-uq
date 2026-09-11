# SPARC-UQ

[Paper (ECCV 2026)](https://arxiv.org/abs/2608.20802) |
[Project](https://julianteusch.github.io/projects/sparc/) |
[Illustrated explainer](https://huggingface.co/spaces/Setoka/SPARC) |
[Visual notebook](notebooks/01_forecasts_and_diagnostics.ipynb) |
[Citation](#citation)

**Keep your point estimator. Add feature-aware scales and held-out prediction intervals.**

A small PyTorch package implementing the reusable scaling core of
[SPARC: Single-Pass Scaling for Motion Forecasting with Conformal Bayesian Last Layers](https://arxiv.org/abs/2608.20802).

**Status: initial integration library.** This is not the full motion benchmark
release and does not yet reproduce the paper's learned graph-temporal covariance
head, stabilized variants or published result tables. No checkpoints or datasets
are included. The distribution name is local; it has not been published to PyPI.

## What It Does

1. Reuse a frozen deterministic predictor and expose its design features.
2. Fit horizon-wise feature precision on a model-fit split.
3. Scale a supplied covariance (or a simple fitted diagonal residual scale).
4. Calibrate on a separate held-out split, then predict in one model forward pass.

The point predictions remain unchanged. No Monte Carlo sampling or retraining of
the point estimator is performed by this package.

```text
point predictor + features -> kappa -> scaled covariance -> conformal intervals
       unchanged              ^                              ^
                         fit split                    held-out calibration
```

## Install and Run

From this repository, using Python 3.10+ and PyTorch 2.2+:

```bash
git clone https://github.com/julianteusch/sparc-uq.git
cd sparc-uq
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/quickstart.py
python examples/structured_forecast.py
python -m pytest
```

The examples are CPU-only, use synthetic data and need no downloads. The first
fits an ordinary linear point estimator, attaches SPARC, evaluates intervals and
checks a save/load round trip. The second demonstrates a custom motion adapter
with manually specified graph-temporal covariance, not a learned paper model.

## Visual Notebook and Diagnostics

The [executed Jupyter notebook](notebooks/01_forecasts_and_diagnostics.ipynb) walks
through the structured synthetic example with forecasts, interval coverage/width,
feature-support projections, covariance comparisons and leverage diagnostics.

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/01_forecasts_and_diagnostics.ipynb
```

Reusable Matplotlib helpers live in `sparc.visualization`; install only
`.[visualization]` when Jupyter is not needed. `sparc.explain_leverage` provides
signed contributions to kappa-1 and local design-feature sensitivities without
plotting dependencies. These are **not causal importance or SHAP values**.
See the [visualization guide](docs/visualization.md) for contracts and limitations.

## Attach an Existing PyTorch Model

For a model whose actual output is a standard `nn.Linear` layer:

```python
from sparc import SPARC, LinearLastLayerAdapter

# model and model.output_layer are your existing, pretrained modules.
adapter = LinearLastLayerAdapter(model, model.output_layer)
uq = SPARC(adapter, model_id="my-checkpoint-v1/preprocessing-v1", alpha=0.05)

# Each iterable yields (inputs, targets). Targets must be [batch, horizon, channel].
# For ordinary vector regression [batch, channel], add targets[:, None, :].
uq.fit(fit_loader)
uq.calibrate(calibration_loader)
prediction = uq.predict(test_inputs)

prediction.mean  # original point forecasts [B,H,C]
prediction.kappa  # feature-dependent scale [B,H]
prediction.std  # scaled model standard deviation, NOT conformal width
prediction.lower, prediction.upper  # calibrated intervals [B,H,C]

uq.save("uncertainty.pt")
restored = SPARC.load("uncertainty.pt", adapter, model_id="my-checkpoint-v1/preprocessing-v1")
```

The adapter collects the linear input during one forward call, augments the bias
feature, temporarily uses evaluation mode and restores module training flags.
It rejects postprocessed outputs, reused final layers and non-tensor returns.
It is not thread-safe for concurrent calls on the same model. Use the explicit
adapter contract below for Transformers, DCT forecasters and custom output heads.

## Custom Predictors and Structured Covariance

Any callable may return a `Forecast`; it does not have to be an `nn.Module`:

```python
from sparc import Forecast, SPARC


def adapter(x):
    mean, design, factor = existing_model.predict_with_features(x)
    return Forecast(mean=mean, features=design, covariance_factor=factor)


uq = SPARC(adapter, model_id="my-structured-model-v1")
```

`predict_with_features` above is a user-defined interface, not a required model
method or an automatically discovered API. See the executable
[structured example](examples/structured_forecast.py) for a complete callable.

| Quantity | Exact Shape | Meaning |
| --- | --- | --- |
| `mean` | `[B,H,C]` | Frozen point forecast |
| `features` | `[B,H,P]` | Explicit design; include the correct bias if applicable |
| `std`, optional | `[B,H,C]` | Positive base standard deviation, before kappa/CP |
| `covariance_factor`, optional | `[B,H*C,H*C]` | Base factor L with covariance L L^T; horizon-major |
| `targets` | `[B,H,C]` | Same units and shape as mean; broadcasting is rejected |

Supply `std` **or** `covariance_factor`, never both. If neither is supplied,
`fit` estimates per-coordinate RMS residuals with a positive floor. This diagonal,
input-independent fallback is intentionally simple; it is **not** the learned
structured covariance in the paper. Supplied covariance heads must already be
trained and frozen. Dense factors are for small problems, not long full-body
trajectories; efficient structured operators remain future work.

For DCT models, `idct_design(features, inverse_dct)` constructs the time-domain
design for a biased linear head shared across coefficients. In particular, its
bias feature is the **row sum of the inverse DCT**, not simply one. It does not
replace your model's normalization, horizon slicing or residual-to-position offset.

## Mathematical Contract

For each horizon t, fitting accumulates

```text
Lambda_t = ridge * I + sum_i phi_t(x_i) phi_t(x_i)^T
kappa_t(x) = 1 + phi_t(x)^T Lambda_t^-1 phi_t(x)
```

Statistics and Cholesky solves use float64. Predictions use the input device;
precision fitting streams batches on the CPU. No explicit inverse is needed.
CPU-stored factors are currently transferred to the input device per prediction;
this initial version is not a latency-optimized GPU implementation.

For full covariance, rows of L are multiplied by sqrt(kappa) in horizon-major
order: `L_pred = D L`, hence `Sigma_pred = D Sigma_base D`. This preserves
correlations, but not necessarily eigenvectors when scales differ across time.
The conformal multiplier defines intervals separately; it is not folded into the
model covariance or represented as a calibrated joint probability density.

In the appropriate conjugate linear model, the predictive covariance has a base
term plus last-layer epistemic inflation. With generic features or a separately
fitted plug-in residual covariance, this library provides **leverage-based scaling**,
not a claim of exact Bayesian inference for arbitrary blackbox predictors.

## Coverage and Data Splits

The default `scope="marginal"` fits a separate quantile for every fixed horizon
and coordinate, without pooling coordinates or overlapping windows. Under
exchangeability of calibration and test rows and a fixed score function, this
targets marginal coverage, not coverage of an entire trajectory simultaneously.

`scope="simultaneous"` instead uses the maximum standardized residual in each row
and one shared quantile. This generic extension targets the whole row as one
prediction unit and may be much wider; it is not the paper's reported marginal
protocol. Neither option provides conditional coverage or protection against
arbitrary distribution shift.

Use independent subject/sequence-aware splits where required. Freeze all model,
feature, preprocessing, covariance and hyperparameter decisions **before**
calibration. Do not reuse calibration outcomes to tune ridge or select features.
The API does not know sample IDs and cannot detect leakage for you. Calling `fit`
again invalidates calibration. Keep the predictor unchanged afterwards.

The finite-sample quantile is the ceil((n+1)(1-alpha))-th ordered score, with an
additional infinity atom. Too few calibration rows correctly yield infinite
intervals rather than a falsely finite guarantee. For 95% coverage, at least 19
rows are needed for a potentially finite quantile; this is not a recommendation
that 19 rows are sufficient for stable estimates.

## Persistence, Testing and Scope

Save/load serializes only fitted statistics, configuration and quantiles with a
versioned format and `torch.load(weights_only=True)`. The predictor and its
preprocessing must be restored separately. `model_id` is a caller-maintained
compatibility label, not a cryptographic weight check. Load only trusted artifacts.

- [Tests](tests/): closed-form identities, source-function parity, finite-sample
  quantiles, unchanged means, one model call, covariance correlation preservation,
  shape failures, streaming, serialization, synthetic coverage and optional CUDA.
- [Scientific scope and provenance](docs/method-and-provenance.md).
- [Integration guide and limitations](docs/integration.md).
- [Release checklist](docs/release-checklist.md).

## Citation

If you use SPARC or build on its uncertainty-scaling approach in your research,
please cite our paper:

```bibtex
@article{hossain2026sparc,
  title={SPARC: Single-Pass Scaling for Motion Forecasting with Conformal Bayesian Last Layers},
  author={Hossain, Sakif and Teusch, Julian and M{\"u}ller, J{\"o}rg P.},
  journal={arXiv preprint arXiv:2608.20802},
  year={2026},
  doi={10.48550/arXiv.2608.20802},
  url={https://arxiv.org/abs/2608.20802}
}
```

Accepted at ECCV 2026. Sakif Hossain and Julian Teusch contributed equally.

## License

The original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
Dependencies retain their respective licenses. This does not license external
datasets, checkpoints or paper assets that are not distributed in this repository.

The research citation request above is separate from the license terms.
