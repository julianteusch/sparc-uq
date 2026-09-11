import importlib.util
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="Install the visualization extra"
)


@pytest.fixture
def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sparc import visualization

    yield plt, visualization
    plt.close("all")


@pytest.fixture
def prediction():
    from sparc import Prediction

    generator = torch.Generator().manual_seed(7)
    mean = torch.randn(10, 4, 2, generator=generator)
    return Prediction(
        mean=mean,
        std=torch.ones_like(mean),
        kappa=torch.ones(10, 4) * 1.2,
        lower=mean - 1,
        upper=mean + 1,
    ), mean.clone()


def test_core_import_does_not_load_matplotlib():
    result = subprocess.run(
        [sys.executable, "-c", "import sys,sparc; assert 'matplotlib' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_forecast_preserves_input_and_draws_nonblank_pixels(plotting, prediction, tmp_path):
    plt, viz = plotting
    p, target = prediction
    original = p.mean.clone()
    fig, ax = plt.subplots(layout="constrained")
    assert viz.plot_forecast(p, target, ax=ax) is ax
    fig.canvas.draw()
    import numpy as np

    pixels = np.asarray(fig.canvas.buffer_rgba())
    assert pixels[..., :3].std() > 5
    assert len(ax.lines) == 2 and len(ax.collections) == 1
    torch.testing.assert_close(p.mean, original)
    fig.savefig(tmp_path / "forecast.svg")
    assert (tmp_path / "forecast.svg").stat().st_size > 1000


def test_calibration_panel_values(plotting, prediction):
    _, viz = plotting
    p, y = prediction
    axes = viz.plot_calibration(p, y, nominal=0.95)
    assert list(axes[0].lines[0].get_ydata()) == [1.0, 1.0, 1.0, 1.0]
    assert list(axes[1].lines[0].get_ydata()) == pytest.approx([2.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        viz.plot_calibration(p, y, nominal=1)
    with pytest.raises(ValueError):
        viz.plot_calibration(p, y[:, 0], nominal=0.95)


def test_covariance_scales_and_normalization(plotting):
    _, viz = plotting
    matrix = torch.tensor([[4.0, 1.0], [1.0, 1.0]])
    ax = viz.plot_covariance(matrix, correlation=True)
    torch.testing.assert_close(
        torch.tensor(ax.images[0].get_array().tolist()), torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    )
    assert ax.images[0].get_clim() == (-1.0, 1.0)
    ax = viz.plot_covariance(matrix, vmin=-8, vmax=8)
    assert ax.images[0].get_clim() == (-8, 8)
    with pytest.raises(ValueError):
        viz.plot_covariance(torch.ones(2, 3))
    with pytest.raises(ValueError):
        viz.plot_covariance(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        viz.plot_covariance(torch.tensor([[1.0, 2.0], [2.0, 1.0]]))


def test_signed_top_k_includes_remainder(plotting):
    from sparc import FeatureLeverage, explain_leverage

    _, viz = plotting
    leverage = FeatureLeverage().fit([torch.ones(30, 1, 3)])
    explanation = explain_leverage(leverage, torch.tensor([[[1.0, 2.0, 3.0]]]))
    ax = viz.plot_leverage_terms(explanation, top_k=1)
    assert len(ax.patches) == 2
    assert sum(patch.get_width() for patch in ax.patches) == pytest.approx(
        explanation.kappa.item() - 1
    )
    assert "Other features" in ax.get_yticklabels()[1].get_text()
    with pytest.raises(ValueError):
        viz.plot_leverage_terms(explanation, feature_names=["wrong"])


def test_risk_and_feature_projection(plotting, prediction):
    _, viz = plotting
    p, y = prediction
    ax = viz.plot_risk_diagnostic(p, y)
    assert ax.collections[0].get_offsets().shape == (10, 2)
    fit, query = torch.ones(20, 3), torch.zeros(5, 3)
    ax = viz.plot_feature_support(fit, query, torch.ones(5), dimensions=(1, 2))
    assert len(ax.collections) == 2
    with pytest.raises(ValueError):
        viz.plot_feature_support(fit, query, torch.ones(4))
    with pytest.raises(ValueError):
        viz.plot_feature_support(fit, query, torch.ones(5), dimensions=(1, 1))


def test_invalid_intervals_and_indices_are_not_silently_clipped(plotting, prediction):
    _, viz = plotting
    p, _ = prediction
    with pytest.raises(ValueError, match="Calibrated"):
        viz.plot_forecast(replace(p, lower=None, upper=None))
    with pytest.raises(ValueError, match="infinite"):
        viz.plot_forecast(replace(p, upper=torch.full_like(p.mean, float("inf"))))
    for kwargs in [{"sample": -1}, {"channel": 2}, {"times": [1, 1, 2, 3]}]:
        with pytest.raises(ValueError):
            viz.plot_forecast(p, **kwargs)
