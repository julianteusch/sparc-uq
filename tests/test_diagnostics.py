import pytest
import torch

from sparc import FeatureLeverage, explain_leverage


def test_signed_terms_and_design_gradient():
    generator = torch.Generator().manual_seed(11)
    train = torch.randn(40, 3, 4, generator=generator, dtype=torch.float64)
    query = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
    leverage = FeatureLeverage(0.2).fit([train])
    before = leverage.cholesky.clone()
    explanation = explain_leverage(leverage, query)
    torch.testing.assert_close(explanation.kappa, leverage(query))
    torch.testing.assert_close(explanation.terms.sum(-1), explanation.kappa - 1)
    epsilon = 1e-5
    for feature in range(4):
        offset = torch.zeros_like(query)
        offset[..., feature] = epsilon
        numerical = (leverage(query + offset) - leverage(query - offset)) / (2 * epsilon)
        torch.testing.assert_close(
            numerical, explanation.sensitivity[..., feature], atol=1e-9, rtol=1e-6
        )
    torch.testing.assert_close(leverage.cholesky, before)
    assert not explanation.terms.requires_grad


def test_correlated_design_can_have_negative_terms():
    leverage = FeatureLeverage().fit([torch.ones(100, 1, 2)])
    explanation = explain_leverage(leverage, torch.tensor([[[1.0, 2.0]]]))
    assert explanation.terms[0, 0, 0] < 0
    assert explanation.terms[0, 0, 1] > 0
    torch.testing.assert_close(explanation.terms.sum(-1), explanation.kappa - 1)


def test_explanation_requires_fitted_compatible_design():
    leverage = FeatureLeverage()
    with pytest.raises(RuntimeError):
        explain_leverage(leverage, torch.ones(3, 1, 2))
    leverage.fit([torch.ones(3, 1, 2)])
    with pytest.raises(ValueError):
        explain_leverage(leverage, torch.ones(3, 1, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_explanation_cuda():
    leverage = FeatureLeverage().fit([torch.ones(30, 1, 2)])
    query = torch.tensor([[[1.0, 2.0]]], device="cuda")
    explanation = explain_leverage(leverage, query)
    assert explanation.terms.is_cuda
    torch.testing.assert_close(explanation.terms.sum(-1), explanation.kappa - 1)
