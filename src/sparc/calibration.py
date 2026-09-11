import math

import torch

from ._validation import cpu64, tensor


class SplitConformal:
    """Absolute standardized residual scores; independent rows are the units.

    marginal: one quantile per fixed (horizon, coordinate), never pooled.
    simultaneous: one maximum score per row, covers the entire output vector.
    Both require exchangeability and a fixed scoring function before calibration.
    """

    def __init__(self, alpha: float = 0.05, scope: str = "marginal"):
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if scope not in {"marginal", "simultaneous"}:
            raise ValueError("scope must be marginal or simultaneous")
        self.alpha, self.scope = float(alpha), scope
        self.q = None
        self.n_calibration = 0
        self.output_shape = None

    def fit_scores(self, scores: torch.Tensor):
        scores = cpu64(tensor(scores, "scores [N,H,C]", 3))
        if (scores < 0).any():
            raise ValueError("Scores must be nonnegative")
        output_shape = tuple(scores.shape[1:])
        if self.scope == "simultaneous":
            scores = scores.amax(dim=(1, 2), keepdim=True)
        n = scores.shape[0]
        order = math.ceil((n + 1) * (1 - self.alpha))
        # Include the conformal +infinity atom; clamping to n is anti-conservative.
        q = (
            torch.full(scores.shape[1:], float("inf"), dtype=torch.float64)
            if order > n
            else scores.kthvalue(order, dim=0).values
        )
        self.q, self.n_calibration, self.output_shape = q, n, output_shape
        return self

    def radius(self, std: torch.Tensor) -> torch.Tensor:
        if self.q is None:
            raise RuntimeError("Calibrate before requesting prediction intervals")
        std = tensor(std, "std [N,H,C]", 3)
        if tuple(std.shape[1:]) != self.output_shape or (std <= 0).any():
            raise ValueError("std must be positive and match the calibration shape")
        return std * self.q.to(std.device)
