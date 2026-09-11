import math

import pytest
import torch
from torch import nn

from sparc import (
    SPARC,
    FeatureLeverage,
    Forecast,
    LinearLastLayerAdapter,
    SplitConformal,
    graph_temporal_factor,
    idct_design,
    inflate_factor,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(162)
    torch.set_num_threads(2)


def sample(n=40):
    return torch.randn(n, 3), torch.randn(n, 1, 2)


def wrapper(**kwargs):
    model = nn.Linear(3, 2)
    return SPARC(LinearLastLayerAdapter(model, model), model_id="test-v1", **kwargs)


def test_leverage_matches_closed_form_and_streaming():
    phi = torch.randn(60, 4, 7, dtype=torch.float64)
    test = torch.randn(10, 4, 7, dtype=torch.float64)
    a = FeatureLeverage(0.3).fit([phi])
    b = FeatureLeverage(0.3).fit(phi.split(13))
    precision = torch.einsum("nhp,nhq->hpq", phi, phi) + 0.3 * torch.eye(7)
    expected = 1 + torch.einsum("nhp,hpq,nhq->nh", test, precision.inverse(), test)
    torch.testing.assert_close(a(test), expected)
    torch.testing.assert_close(b(test), expected)
    assert a.n_fit == 60
    assert (a(test) >= 1).all()


def test_rank_deficient_features_are_regularized():
    phi = torch.ones(5, 2, 8)
    kappa = FeatureLeverage().fit([phi])(phi)
    assert torch.isfinite(kappa).all()


@pytest.mark.parametrize("ridge", [0, -1, float("nan"), float("inf")])
def test_invalid_ridge(ridge):
    with pytest.raises(ValueError):
        FeatureLeverage(ridge)


def test_leverage_invalid_inputs():
    model = FeatureLeverage()
    with pytest.raises(RuntimeError):
        model(torch.ones(2, 1, 3))
    for batches in [
        [],
        [torch.zeros(0, 1, 3)],
        [torch.ones(2, 3)],
        [torch.full((2, 1, 3), float("nan"))],
        [torch.ones(2, 1, 3), torch.ones(2, 2, 3)],
    ]:
        with pytest.raises(ValueError):
            model.fit(batches)
    model.fit([torch.ones(2, 1, 3)])
    with pytest.raises(ValueError):
        model(torch.ones(2, 2, 3))


def test_finite_sample_quantile_no_interpolation():
    scores = torch.arange(1.0, 20.0)[:, None, None]
    cp = SplitConformal(alpha=0.05).fit_scores(scores)
    assert cp.q.item() == 19
    assert torch.isinf(SplitConformal(alpha=0.01).fit_scores(scores).q).all()
    assert SplitConformal(alpha=0.2).fit_scores(scores).q.item() == 16


def test_calibration_scope_and_no_coordinate_pooling():
    scores = torch.stack([torch.arange(1.0, 20.0), torch.arange(1.0, 20.0) * 10], -1)[:, None]
    marginal = SplitConformal().fit_scores(scores)
    joint = SplitConformal(scope="simultaneous").fit_scores(scores)
    torch.testing.assert_close(marginal.q, torch.tensor([[19.0, 190.0]], dtype=torch.float64))
    assert joint.q.shape == (1, 1)
    assert joint.q.item() == 190
    with pytest.raises(ValueError):
        marginal.radius(torch.ones(2, 2, 1))


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0}, {"alpha": 1}, {"alpha": math.nan}, {"scope": "pooled"}]
)
def test_invalid_calibration_config(kwargs):
    with pytest.raises(ValueError):
        SplitConformal(**kwargs)


def test_scalar_bayesian_predictive_variance():
    phi = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float64)
    kappa = FeatureLeverage(2).fit([phi])(torch.tensor([[[3.0]]]))
    # Bayesian linear regression with weight prior variance sigma^2/lambda.
    sigma2 = 4.0
    expected = sigma2 + 9 * sigma2 / (2 + 1 + 4)
    assert (sigma2 * kappa).item() == pytest.approx(expected)


def test_adapter_one_forward_and_unchanged_model():
    model = nn.Sequential(nn.Linear(3, 8), nn.Dropout(0.7), nn.Linear(8, 2)).train()
    model[0].eval()
    before = [m.training for m in model.modules()]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    adapter = LinearLastLayerAdapter(model, model[-1])
    x = torch.randn(7, 3)
    first, second = adapter(x), adapter(x)
    handle.remove()
    assert len(calls) == 2
    assert [m.training for m in model.modules()] == before
    assert not model[-1]._forward_hooks
    assert first.features.shape == (7, 1, 9)
    torch.testing.assert_close(first.mean, second.mean)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    torch.testing.assert_close(
        first.mean, first.features[..., :-1] @ model[-1].weight.T + model[-1].bias
    )


def test_adapter_rejects_postprocessed_output_and_removes_hook():
    model = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
    adapter = LinearLastLayerAdapter(model, model[0])
    with pytest.raises(ValueError, match="unchanged"):
        adapter(torch.ones(4, 3))
    assert not model[0]._forward_hooks


def test_idct_design_reconstructs_biased_shared_head():
    features = torch.randn(4, 5, 3)
    basis = torch.randn(7, 5)
    layer = nn.Linear(3, 2)
    design = idct_design(features, basis)
    expected = torch.einsum("hk,bkc->bhc", basis, layer(features))
    weights = torch.cat([layer.weight.T, layer.bias[None]], 0)
    torch.testing.assert_close(design @ weights, expected)


def test_graph_covariance_and_inflation_preserve_correlations():
    temporal = torch.tensor([[[1.0, 0.0], [0.4, 0.8]]], dtype=torch.float64)
    adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    factor = graph_temporal_factor(temporal, adjacency, tau=2, epsilon=0.2, coordinates=1)
    precision = torch.tensor([[2.2, -2.0], [-2.0, 2.2]], dtype=torch.float64)
    expected = torch.kron(temporal[0] @ temporal[0].T, precision.inverse().contiguous())
    cov = factor @ factor.mT
    torch.testing.assert_close(cov[0], expected)
    scaled = inflate_factor(factor, torch.tensor([[1.0, 9.0]]), 2)
    inflated = scaled @ scaled.mT

    def correlation(x):
        std = x.diagonal(dim1=-2, dim2=-1).sqrt()
        return x / std.unsqueeze(-1) / std.unsqueeze(-2)

    torch.testing.assert_close(correlation(cov), correlation(inflated))
    assert (torch.linalg.eigvalsh(inflated) > 0).all()


def test_fit_calibrate_roundtrip_and_refit_invalidation(tmp_path):
    uq = wrapper()
    fit, cal, test = sample(), sample(), sample()
    uq.fit([fit]).calibrate([cal])
    before = uq.leverage.cholesky.clone()
    p = uq.predict(test[0])
    torch.testing.assert_close(p.mean, uq.predictor(test[0]).mean)
    uq.calibrate([cal])
    torch.testing.assert_close(uq.leverage.cholesky, before)
    path = tmp_path / "uq.pt"
    uq.save(path)
    restored = SPARC.load(path, uq.predictor, model_id="test-v1")
    torch.testing.assert_close(p.lower, restored.predict(test[0]).lower)
    with pytest.raises(ValueError, match="model_id"):
        SPARC.load(path, uq.predictor, model_id="other")
    uq.fit([fit])
    with pytest.raises(RuntimeError, match="Calibrate"):
        uq.predict(test[0])
    assert uq.predict(test[0], intervals=False).lower is None


def test_prediction_calls_model_once():
    uq = wrapper().fit([sample()]).calibrate([sample()])
    calls = []
    handle = uq.predictor.model.register_forward_hook(lambda *args: calls.append(1))
    uq.predict(torch.ones(3, 3))
    handle.remove()
    assert len(calls) == 1


def test_bad_data_and_failed_refit_preserve_previous_state():
    uq = wrapper()
    with pytest.raises(RuntimeError):
        uq.predict(torch.ones(2, 3))
    uq.fit([sample()]).calibrate([sample()])
    chol, q = uq.leverage.cholesky.clone(), uq.calibration.q.clone()
    for batches in [[], [(torch.ones(4, 3), torch.ones(4, 2))]]:
        with pytest.raises(ValueError):
            uq.fit(batches)
    torch.testing.assert_close(chol, uq.leverage.cholesky)
    torch.testing.assert_close(q, uq.calibration.q)
    with pytest.raises(ValueError):
        uq.calibrate([])


@pytest.mark.parametrize("mode", ["std", "factor"])
def test_supplied_covariance(mode, tmp_path):
    def adapter(x):
        kwargs = (
            {"std": torch.ones_like(x)[:, None]}
            if mode == "std"
            else {"covariance_factor": torch.eye(2).expand(len(x), -1, -1)}
        )
        return Forecast(x[:, None], x[:, None], **kwargs)

    uq = SPARC(adapter, model_id="provided").fit([(torch.ones(40, 2), torch.ones(40, 1, 2))])
    uq.calibrate([(torch.ones(40, 2), torch.zeros(40, 1, 2))])
    prediction = uq.predict(torch.ones(4, 2))
    torch.testing.assert_close(prediction.std, prediction.kappa.sqrt()[..., None].expand(-1, -1, 2))
    if mode == "factor":
        torch.testing.assert_close(
            prediction.covariance_factor.square().sum(-1).sqrt()[:, None], prediction.std
        )
    path = tmp_path / "supplied.pt"
    uq.save(path)
    restored = SPARC.load(path, adapter, model_id="provided")
    torch.testing.assert_close(prediction.upper, restored.predict(torch.ones(4, 2)).upper)


def test_zero_residual_floor_and_small_calibration_infinite_intervals(tmp_path):
    def adapter(x):
        return Forecast(x[:, None], x[:, None])

    uq = SPARC(adapter, model_id="zero").fit([(torch.ones(3, 2), torch.ones(3, 1, 2))])
    uq.calibrate([(torch.ones(3, 2), torch.ones(3, 1, 2))])
    p = uq.predict(torch.ones(2, 2))
    assert torch.isfinite(p.std).all() and (p.std > 0).all()
    assert torch.isneginf(p.lower).all() and torch.isposinf(p.upper).all()
    uq.save(tmp_path / "small.pt")
    restored = SPARC.load(tmp_path / "small.pt", adapter, model_id="zero")
    assert torch.isposinf(restored.predict(torch.ones(2, 2)).upper).all()


def test_synthetic_coverage_is_reasonable():
    def adapter(x):
        return Forecast(x[:, None], torch.cat([x, torch.ones_like(x)], dim=-1)[:, None])

    def data(n):
        x = torch.randn(n, 1)
        return x, (x + torch.randn(n, 1) * 0.5)[:, None]

    uq = SPARC(adapter, model_id="coverage").fit([data(600)]).calibrate([data(2000)])
    x, y = data(6000)
    p = uq.predict(x)
    coverage = ((y >= p.lower) & (y <= p.upper)).float().mean().item()
    assert 0.92 < coverage < 0.98


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_prediction():
    model = nn.Linear(3, 2).cuda()
    adapter = LinearLastLayerAdapter(model, model)
    x, y = sample()
    uq = SPARC(adapter, model_id="gpu").fit([(x.cuda(), y)]).calibrate([(x.cuda(), y)])
    p = uq.predict(x.cuda())
    assert p.mean.is_cuda and p.lower.is_cuda


def test_saved_state_validation(tmp_path):
    uq = wrapper().fit([sample()]).calibrate([sample()])
    path = tmp_path / "uq.pt"
    uq.save(path)
    state = torch.load(path, weights_only=True)
    state["cholesky"][0, 0, 0] = -1
    torch.save(state, path)
    with pytest.raises(ValueError, match="fitted state"):
        SPARC.load(path, uq.predictor, model_id="test-v1")


def test_simultaneous_roundtrip(tmp_path):
    uq = wrapper(scope="simultaneous").fit([sample()]).calibrate([sample()])
    path = tmp_path / "joint.pt"
    uq.save(path)
    loaded = SPARC.load(path, uq.predictor, model_id="test-v1")
    x, _ = sample()
    torch.testing.assert_close(uq.predict(x).upper, loaded.predict(x).upper)


def test_adapter_restores_state_when_model_raises():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(3, 2)

        def forward(self, x):
            self.head(x)
            raise RuntimeError("deliberate failure")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="deliberate"):
        LinearLastLayerAdapter(model, model.head)(torch.ones(4, 3))
    assert model.training and model.head.training
    assert not model.head._forward_hooks


def test_covariance_validation():
    for adjacency in [torch.tensor([[0.0, 1.0], [0.0, 0.0]]), -torch.eye(2)]:
        with pytest.raises(ValueError):
            graph_temporal_factor(torch.eye(2)[None], adjacency)
    with pytest.raises(ValueError):
        inflate_factor(torch.eye(4)[None], torch.tensor([[1.0, -1.0]]), 2)

    def bad(x):
        return Forecast(x[:, None], x[:, None], covariance_factor=torch.zeros(len(x), 2, 2))

    with pytest.raises(ValueError, match="nonsingular"):
        SPARC(bad, model_id="bad").fit([(torch.ones(4, 2), torch.ones(4, 1, 2))])
