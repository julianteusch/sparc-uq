from collections.abc import Iterable

import torch

from ._validation import cpu64, positive, tensor


class FeatureLeverage:
    """Horizon-wise precision fit from explicit design vectors [N,H,P].

    Bias augmentation and feature normalization belong to the adapter. Statistics
    are accumulated in CPU float64; prediction runs on the input device in float64.
    """

    def __init__(self, ridge: float = 1.0):
        self.ridge = positive(ridge, "ridge")
        self.cholesky = None
        self.n_fit = 0

    @torch.no_grad()
    def fit(self, batches: Iterable[torch.Tensor]):
        gram, count = None, 0
        for features in batches:
            phi = cpu64(tensor(features, "features [N,H,P]", 3))
            update = torch.einsum("nhp,nhq->hpq", phi, phi)
            if gram is not None and update.shape != gram.shape:
                raise ValueError("Feature shape changed between fit batches")
            gram = update if gram is None else gram + update
            count += phi.shape[0]
        if gram is None:
            raise ValueError("Fit requires at least one nonempty batch")
        factor = torch.linalg.cholesky(
            gram + self.ridge * torch.eye(gram.shape[-1], dtype=gram.dtype)
        )
        self.cholesky, self.n_fit = factor, count
        return self

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.cholesky is None:
            raise RuntimeError("Fit feature statistics before prediction")
        phi = tensor(features, "features [N,H,P]", 3).to(dtype=torch.float64)
        if phi.shape[1:] != self.cholesky.shape[:2]:
            raise ValueError("Features do not match fitted horizon and design dimension")
        # Solving L z = phi gives phi^T Lambda^-1 phi = ||z||^2.
        solved = torch.linalg.solve_triangular(
            self.cholesky.to(phi.device), phi.permute(1, 2, 0), upper=False
        )
        return tensor(1 + solved.square().sum(dim=1).T, "kappa", 2)
