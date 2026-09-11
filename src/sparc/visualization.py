"""Optional Matplotlib diagnostics. Import explicitly; core SPARC needs no plotting stack.

All helpers return axes, accept caller-owned axes and never call show(), change
global rcParams, select a backend, fit a model or move its state. Tensor copies
for plotting are detached and transferred to the CPU. Export with ax.figure.savefig.
"""

import math

import torch

from ._validation import cpu64, tensor
from .diagnostics import LeverageExplanation
from .model import Prediction

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as error:
    raise ImportError(
        'From the repository, install plotting with: pip install -e ".[visualization]"'
    ) from error

GREEN = "#008c4f"
ORANGE = "#c35d1c"
BLUE = "#147cad"
GRAY = "#657279"


def _array(value):
    # Avoid coupling plotting to PyTorch's NumPy binary bridge (older torch/newer numpy).
    return np.asarray(cpu64(value).tolist(), dtype=float)


def _axis(ax, *, size=(7, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=size, layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)
    return ax


def _index(value, size, name):
    if not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in [0, {size})")


def _names(names, size):
    if names is None:
        return [f"feature {i}" for i in range(size)]
    if len(names) != size or not all(isinstance(name, str) for name in names):
        raise ValueError("Provide one feature name per design dimension")
    return list(names)


def _intervals(prediction):
    mean = tensor(prediction.mean, "prediction.mean", 3)
    if prediction.lower is None or prediction.upper is None:
        raise ValueError("Calibrated intervals are required for this plot")
    lower, upper = prediction.lower.detach(), prediction.upper.detach()
    if lower.shape != mean.shape or upper.shape != mean.shape:
        raise ValueError("Interval bounds must match the mean shape")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("Cannot plot infinite bounds; check calibration sample size and alpha")
    lower, upper = cpu64(lower), cpu64(upper)
    if (lower > upper).any():
        raise ValueError("Lower bounds exceed upper bounds")
    return cpu64(mean), lower, upper


def _target(target, shape):
    target = cpu64(tensor(target, "target [B,H,C]", 3))
    if target.shape != shape:
        raise ValueError("Target shape must match prediction.mean without broadcasting")
    return target


def plot_forecast(prediction: Prediction, target=None, *, sample=0, channel=0, times=None, ax=None):
    """One coordinate's mean and calibrated interval across the forecast horizon.

    The shaded area joins discrete bounds for readability; it does not assert
    simultaneous trajectory coverage or coverage at unobserved intermediate times.
    Infinite bounds are rejected rather than silently clipped.
    """
    mean, lower, upper = _intervals(prediction)
    _index(sample, mean.shape[0], "sample")
    _index(channel, mean.shape[2], "channel")
    x = np.arange(1, mean.shape[1] + 1) if times is None else np.asarray(times, dtype=float)
    if x.shape != (mean.shape[1],) or not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        raise ValueError("times must be finite, strictly increasing and match the horizon")
    ax = _axis(ax)
    ax.fill_between(
        x,
        _array(lower[sample, :, channel]),
        _array(upper[sample, :, channel]),
        color=GREEN,
        alpha=0.18,
        label="Conformal interval",
    )
    ax.plot(x, _array(mean[sample, :, channel]), color=GREEN, lw=2, label="Frozen point forecast")
    if target is not None:
        y = _target(target, mean.shape)
        ax.plot(
            x,
            _array(y[sample, :, channel]),
            "o--",
            color=ORANGE,
            ms=4,
            lw=1.4,
            label="Observed future",
        )
    ax.set(
        xlabel="Horizon step" if times is None else "Forecast time",
        ylabel="Output value",
        title=f"Sample {sample}, coordinate {channel}",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_calibration(prediction: Prediction, target, *, nominal, axes=None):
    """Empirical mean marginal coverage and average interval width per horizon.

    Coordinates are averaged for visualization only, not pooled for CP fitting.
    No binomial confidence band is drawn: within-row coordinates may be dependent.
    nominal is supplied explicitly by the caller and must match their calibration.
    """
    if not math.isfinite(nominal) or not 0 < nominal < 1:
        raise ValueError("nominal must be between zero and one")
    mean, lower, upper = _intervals(prediction)
    y = _target(target, mean.shape)
    coverage = ((y >= lower) & (y <= upper)).double().mean(dim=(0, 2))
    width = (upper - lower).mean(dim=(0, 2))
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    if len(axes) != 2:
        raise ValueError("Provide exactly two axes")
    left, right = [_axis(ax) for ax in axes]
    x = np.arange(1, mean.shape[1] + 1)
    left.plot(x, _array(coverage), "o-", color=GREEN, lw=2, label="Held-out rows")
    left.axhline(nominal, color=ORANGE, ls="--", label=f"Nominal {nominal:.0%}")
    left.set(
        xlabel="Horizon step",
        ylabel="Mean marginal coverage",
        ylim=(0, 1.02),
        title="Empirical coverage",
    )
    left.legend(frameon=False, fontsize=9)
    right.plot(x, _array(width), "o-", color=BLUE, lw=2)
    right.set(xlabel="Horizon step", ylabel="Mean interval width", title="Interval efficiency")
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.18)
    return axes


def plot_covariance(matrix, *, correlation=False, vmin=None, vmax=None, ax=None):
    """A single square covariance matrix, optionally normalized to correlations.

    Pass shared vmin/vmax when comparing covariance panels. Correlation panels
    always use [-1,1] by default. Axes follow the caller's output flattening order.
    """
    matrix = cpu64(tensor(matrix, "covariance matrix", 2))
    if matrix.shape[0] != matrix.shape[1] or not torch.allclose(matrix, matrix.T):
        raise ValueError("Covariance must be square and symmetric")
    if (matrix.diagonal() <= 0).any():
        raise ValueError("Covariance diagonal must be strictly positive")
    tolerance = 1e-6 * matrix.diagonal().max()
    if torch.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError("Covariance must be positive semidefinite")
    if correlation:
        std = matrix.diagonal().sqrt()
        matrix = matrix / std[:, None] / std[None, :]
    limit = 1.0 if correlation else matrix.abs().max().item()
    ax = _axis(ax, size=(4.6, 4))
    artist = ax.imshow(
        _array(matrix),
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-limit if vmin is None else vmin,
        vmax=limit if vmax is None else vmax,
    )
    ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xlabel="Flattened output",
        ylabel="Flattened output",
        title="Correlation" if correlation else "Covariance",
    )
    return ax


def plot_leverage_terms(
    explanation: LeverageExplanation,
    *,
    sample=0,
    horizon=0,
    feature_names=None,
    top_k=None,
    ax=None,
):
    """Signed contributions to kappa-1, not causal importance or SHAP values.

    If top_k is given, remaining signed terms are combined into 'Other features'
    so the displayed bars still sum to the full quadratic form.
    """
    terms = tensor(explanation.terms, "terms", 3)
    _index(sample, terms.shape[0], "sample")
    _index(horizon, terms.shape[1], "horizon")
    values = _array(terms[sample, horizon])
    names = _names(feature_names, len(values))
    order = np.argsort(-np.abs(values), kind="stable")
    count = len(values) if top_k is None else top_k
    if not isinstance(count, int) or not 1 <= count <= len(values):
        raise ValueError("top_k must be between 1 and the design dimension")
    shown, labels = list(values[order[:count]]), [names[i] for i in order[:count]]
    if count < len(values):
        shown.append(float(values[order[count:]].sum()))
        labels.append("Other features (signed sum)")
    ax = _axis(ax, size=(8, max(3.4, len(shown) * 0.38 + 1.4)))
    ax.barh(np.arange(len(shown)), shown, color=[GREEN if v >= 0 else ORANGE for v in shown])
    ax.set_yticks(np.arange(len(shown)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set(
        xlabel=r"Signed contribution to $\kappa-1$",
        title=rf"Design-space decomposition: $\kappa={explanation.kappa[sample, horizon].item():.3f}$",
    )
    return ax


def plot_risk_diagnostic(prediction: Prediction, target, *, ax=None):
    """Observed error versus mean horizon kappa; descriptive, not a safety/OOD test."""
    mean = cpu64(tensor(prediction.mean, "mean", 3))
    y = _target(target, mean.shape)
    kappa = cpu64(tensor(prediction.kappa, "kappa", 2))
    if kappa.shape != mean.shape[:2]:
        raise ValueError("kappa must match mean batch/horizon dimensions")
    error = (y - mean).square().mean(dim=(1, 2)).sqrt()
    ax = _axis(ax)
    ax.scatter(_array(kappa.mean(1)), _array(error), s=16, alpha=0.4, color=BLUE, edgecolors="none")
    ax.set(
        xlabel=r"Mean horizon $\kappa$",
        ylabel="Observed row RMSE",
        title="Risk diagnostic on held-out rows",
    )
    ax.grid(alpha=0.18)
    return ax


def plot_feature_support(
    fit_features, query_features, kappa, *, dimensions=(0, 1), feature_names=None, ax=None
):
    """Two-dimensional projection of a design space, colored by supplied row kappa.

    Inputs are [N,P], [M,P], [M], e.g. slices at one fixed horizon. Distances in
    this projection alone do not explain leverage in the full representation.
    """
    fit = cpu64(tensor(fit_features, "fit_features", 2))
    query = cpu64(tensor(query_features, "query_features", 2))
    scores = cpu64(tensor(kappa, "kappa", 1))
    if query.shape[1] != fit.shape[1] or scores.shape[0] != query.shape[0]:
        raise ValueError("Design dimensions and query score counts must match")
    if len(dimensions) != 2 or dimensions[0] == dimensions[1]:
        raise ValueError("Choose two distinct design dimensions")
    for dim in dimensions:
        _index(dim, fit.shape[1], "dimension")
    names = _names(feature_names, fit.shape[1])
    d0, d1 = dimensions
    ax = _axis(ax)
    ax.scatter(
        _array(fit[:, d0]),
        _array(fit[:, d1]),
        s=10,
        alpha=0.25,
        color=GRAY,
        label="Fit design rows",
        edgecolors="none",
    )
    points = ax.scatter(
        _array(query[:, d0]),
        _array(query[:, d1]),
        c=_array(scores),
        cmap="viridis",
        s=38,
        edgecolors="white",
        linewidths=0.5,
        label="Query design rows",
    )
    ax.figure.colorbar(points, ax=ax, label=r"$\kappa$")
    ax.set(xlabel=names[d0], ylabel=names[d1], title="Two-dimensional design projection")
    ax.legend(frameon=False, fontsize=9)
    return ax
