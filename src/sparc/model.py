from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ._validation import cpu64, positive, tensor
from .calibration import SplitConformal
from .covariance import inflate_factor
from .leverage import FeatureLeverage


@dataclass(frozen=True)
class Forecast:
    """One frozen model evaluation. Axes: batch, horizon, coordinate/design.

    Provide either std [B,H,C], a nonsingular covariance factor [B,HC,HC],
    or neither (SPARC then fits a fixed diagonal residual second moment).
    Supplied scales must be frozen, pre-kappa and pre-conformal quantities.
    """

    mean: torch.Tensor
    features: torch.Tensor
    std: torch.Tensor | None = None
    covariance_factor: torch.Tensor | None = None


@dataclass(frozen=True)
class Prediction:
    mean: torch.Tensor
    std: torch.Tensor
    kappa: torch.Tensor
    lower: torch.Tensor | None
    upper: torch.Tensor | None
    covariance_factor: torch.Tensor | None = None


class SPARC:
    """Wrap a deterministic predictor without changing its point predictions.

    fit() uses a model-fit split, calibrate() a disjoint held-out split. The
    caller must enforce split independence and keep the predictor fixed. The
    API cannot detect duplicated subjects, overlapping windows or changed weights.
    """

    def __init__(
        self,
        predictor: Callable[[Any], Forecast],
        *,
        model_id: str,
        ridge: float = 1.0,
        alpha: float = 0.05,
        scope: str = "marginal",
        min_std: float = 1e-6,
    ):
        if not callable(predictor):
            raise ValueError("predictor must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must identify frozen weights AND preprocessing")
        self.predictor, self.model_id = predictor, model_id
        self.leverage = FeatureLeverage(ridge)
        self.calibration = SplitConformal(alpha, scope)
        self.min_std = positive(min_std, "min_std")
        self._base_std = None
        self._shape = None
        self._mode = None

    def _forecast(self, x: Any) -> tuple[Forecast, str]:
        f = self.predictor(x)
        if not isinstance(f, Forecast):
            raise ValueError("predictor must return Forecast")
        mean = tensor(f.mean, "mean [B,H,C]", 3)
        features = tensor(f.features, "features [B,H,P]", 3)
        if mean.shape[:2] != features.shape[:2] or mean.device != features.device:
            raise ValueError("mean and features must agree in batch/horizon and device")
        if f.std is not None and f.covariance_factor is not None:
            raise ValueError("Supply std OR covariance_factor, not both")
        mode = "residual"
        if f.std is not None:
            std = tensor(f.std, "std", 3)
            if std.shape != mean.shape or std.device != mean.device or (std <= 0).any():
                raise ValueError("std must be positive and match mean shape/device")
            mode = "std"
        if f.covariance_factor is not None:
            factor = tensor(f.covariance_factor, "covariance_factor", 3)
            n, h, c = mean.shape
            if factor.shape != (n, h * c, h * c) or factor.device != mean.device:
                raise ValueError("covariance_factor must have shape [B,HC,HC] on mean device")
            # Factors need not be triangular, but must represent a nondegenerate density.
            sign, logdet = torch.linalg.slogdet(factor.to(torch.float64))
            if (sign == 0).any() or not torch.isfinite(logdet).all():
                raise ValueError("covariance_factor must be nonsingular")
            mode = "factor"
        return f, mode

    @staticmethod
    def _target(y: torch.Tensor, mean: torch.Tensor):
        tensor(y, "target [B,H,C]", 3)
        if y.shape != mean.shape:
            raise ValueError("target must have exactly the mean shape (no broadcasting)")

    @torch.no_grad()
    def fit(self, batches: Iterable[tuple[Any, torch.Tensor]]):
        """Stream fit statistics; a successful refit invalidates all calibration."""
        shape, mode, squares, count = None, None, None, 0

        def designs():
            nonlocal shape, mode, squares, count
            for x, y in batches:
                f, current_mode = self._forecast(x)
                self._target(y, f.mean)
                current_shape = tuple(f.mean.shape[1:])
                if shape is not None and (shape != current_shape or mode != current_mode):
                    raise ValueError("Output shape or covariance mode changed during fit")
                shape, mode = current_shape, current_mode
                if mode == "residual":
                    update = (cpu64(y) - cpu64(f.mean)).square().sum(0)
                    squares = update if squares is None else squares + update
                count += f.mean.shape[0]
                yield f.features

        leverage = FeatureLeverage(self.leverage.ridge).fit(designs())
        base = None if squares is None else (squares / count).sqrt().clamp_min(self.min_std)
        if base is not None:
            tensor(base, "residual scale", 2)
        self.leverage, self._base_std = leverage, base
        self._shape, self._mode = shape, mode
        self.calibration = SplitConformal(self.calibration.alpha, self.calibration.scope)
        return self

    @torch.no_grad()
    def _distribution(self, x: Any) -> tuple[Forecast, torch.Tensor, torch.Tensor, Any]:
        if self._shape is None:
            raise RuntimeError("Fit before prediction or calibration")
        f, mode = self._forecast(x)
        if tuple(f.mean.shape[1:]) != self._shape or mode != self._mode:
            raise ValueError("Predictor output shape/covariance mode differs from fit")
        kappa = self.leverage(f.features)
        factor = None
        if mode == "factor":
            factor = inflate_factor(f.covariance_factor, kappa, self._shape[-1])
            std = torch.linalg.vector_norm(factor, dim=-1).reshape(f.mean.shape)
        else:
            base = self._base_std.to(f.mean.device) if mode == "residual" else f.std
            std = base * kappa.sqrt().unsqueeze(-1)
        tensor(std, "predictive std", 3)
        if (std <= 0).any():
            raise ValueError("Predictive scale underflowed; rescale the problem")
        return f, kappa, std, factor

    @torch.no_grad()
    def calibrate(self, batches: Iterable[tuple[Any, torch.Tensor]]):
        """Use held-out rows only; never fit predictor, features or noise here."""
        scores = []
        for x, y in batches:
            f, _, std, _ = self._distribution(x)
            self._target(y, f.mean)
            scores.append((cpu64(y) - cpu64(f.mean)).abs() / cpu64(std))
        if not scores:
            raise ValueError("Calibration requires at least one nonempty batch")
        calibration = SplitConformal(self.calibration.alpha, self.calibration.scope)
        calibration.fit_scores(torch.cat(scores))
        self.calibration = calibration
        return self

    @torch.no_grad()
    def predict(self, x: Any, *, intervals: bool = True) -> Prediction:
        f, kappa, std, factor = self._distribution(x)
        radius = self.calibration.radius(std) if intervals else None
        return Prediction(
            mean=f.mean.detach(),
            std=std,
            kappa=kappa,
            lower=None if radius is None else f.mean - radius,
            upper=None if radius is None else f.mean + radius,
            covariance_factor=factor,
        )

    def save(self, path: str | Path):
        """Save UQ state only, never pickle the predictor or include training rows."""
        if self._shape is None:
            raise RuntimeError("Nothing fitted to save")
        torch.save(
            {
                "format_version": 1,
                "model_id": self.model_id,
                "ridge": self.leverage.ridge,
                "min_std": self.min_std,
                "alpha": self.calibration.alpha,
                "scope": self.calibration.scope,
                "cholesky": self.leverage.cholesky,
                "n_fit": self.leverage.n_fit,
                "base_std": self._base_std,
                "shape": self._shape,
                "mode": self._mode,
                "q": self.calibration.q,
                "n_calibration": self.calibration.n_calibration,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, predictor: Callable, *, model_id: str):
        """Load trusted UQ state; caller restores identical model/preprocessing.

        model_id is a caller-provided compatibility label, not a weight checksum.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state.get("model_id") != model_id:
            raise ValueError("Unsupported state format or model_id mismatch")
        obj = cls(
            predictor,
            model_id=model_id,
            ridge=state["ridge"],
            min_std=state["min_std"],
            alpha=state["alpha"],
            scope=state["scope"],
        )
        shape = tuple(state["shape"])
        chol = tensor(state["cholesky"], "saved cholesky", 3)
        if (
            len(shape) != 2
            or any(not isinstance(s, int) or s < 1 for s in shape)
            or chol.shape[0] != shape[0]
            or chol.shape[1] != chol.shape[2]
            or not torch.equal(chol, chol.tril())
            or (chol.diagonal(dim1=-2, dim2=-1) <= 0).any()
            or state["n_fit"] < 1
            or state["mode"] not in {"residual", "std", "factor"}
        ):
            raise ValueError("Invalid fitted state")
        base = state["base_std"]
        if state["mode"] == "residual":
            tensor(base, "saved base_std", 2)
            if tuple(base.shape) != shape or (base <= 0).any():
                raise ValueError("Invalid saved residual scale")
        elif base is not None:
            raise ValueError("Unexpected residual scale for supplied covariance")
        q = state["q"]
        if q is not None:
            q_shape = shape if obj.calibration.scope == "marginal" else (1, 1)
            if (
                not isinstance(q, torch.Tensor)
                or not q.is_floating_point()
                or tuple(q.shape) != q_shape
                or torch.isnan(q).any()
                or (q < 0).any()
                or state["n_calibration"] < 1
            ):
                raise ValueError("Invalid calibration state")
        elif state["n_calibration"] != 0:
            raise ValueError("Calibration count without quantiles")
        obj.leverage.cholesky, obj.leverage.n_fit = cpu64(chol), state["n_fit"]
        obj._base_std, obj._shape, obj._mode = base, shape, state["mode"]
        obj.calibration.q = q
        obj.calibration.n_calibration = state["n_calibration"]
        obj.calibration.output_shape = shape if q is not None else None
        return obj
