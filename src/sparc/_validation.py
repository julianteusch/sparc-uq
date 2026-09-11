import math

import torch


def positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != ndim or any(s == 0 for s in value.shape):
        raise ValueError(f"{name} must be nonempty and {ndim}-dimensional")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.detach()


def cpu64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64)
