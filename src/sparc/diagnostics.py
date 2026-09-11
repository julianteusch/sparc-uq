"""Exact diagnostics of feature leverage, not causal model explanations."""

from dataclasses import dataclass

import torch

from ._validation import tensor
from .leverage import FeatureLeverage


@dataclass(frozen=True)
class LeverageExplanation:
    """All arrays use the original design basis, including any bias coordinate.

    terms [B,H,P] sum to kappa - 1 and can be negative due to cross terms.
    sensitivity [B,H,P] is d(kappa)/d(phi), NOT a gradient with respect to raw
    inputs. Both depend on feature units and basis; neither is SHAP or causal
    feature importance. Fitted precision is held fixed throughout.
    """

    kappa: torch.Tensor
    terms: torch.Tensor
    sensitivity: torch.Tensor


@torch.no_grad()
def explain_leverage(leverage: FeatureLeverage, features: torch.Tensor) -> LeverageExplanation:
    """Decompose phi^T Lambda^-1 phi and compute its local design gradient.

    c_p = phi_p (Lambda^-1 phi)_p; grad_phi(kappa) = 2 Lambda^-1 phi.
    This adds diagnostic solves; it does not change model predictions or fit state.
    """
    kappa = leverage(features)
    phi = features.detach().to(dtype=torch.float64)
    projected = torch.cholesky_solve(
        phi.permute(1, 2, 0), leverage.cholesky.to(phi.device)
    ).permute(2, 0, 1)
    terms = tensor(phi * projected, "leverage terms", 3)
    sensitivity = tensor(2 * projected, "leverage sensitivity", 3)
    return LeverageExplanation(kappa=kappa, terms=terms, sensitivity=sensitivity)
