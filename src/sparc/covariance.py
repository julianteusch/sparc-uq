import torch

from ._validation import positive, tensor


def inflate_factor(factor: torch.Tensor, kappa: torch.Tensor, channels: int) -> torch.Tensor:
    """Return D L for covariance D (L L^T) D; flattening is horizon-major.

    factor: [N,H*C,H*C], kappa: [N,H]. Positive congruence preserves
    correlations, not necessarily covariance eigenvectors. No CP quantile here.
    """
    factor = tensor(factor, "factor", 3)
    kappa = tensor(kappa, "kappa", 2)
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be a positive integer")
    n, h = kappa.shape
    if factor.shape != (n, h * channels, h * channels) or (kappa <= 0).any():
        raise ValueError("Incompatible factor/kappa shape or nonpositive kappa")
    if factor.device != kappa.device:
        raise ValueError("factor and kappa must be on the same device")
    return factor * kappa.sqrt().repeat_interleave(channels, dim=1).unsqueeze(-1)


def graph_temporal_factor(
    temporal_factor: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    tau: float = 1.0,
    epsilon: float = 0.1,
    coordinates: int = 3,
) -> torch.Tensor:
    """Small-problem helper: dense factor of Sigma_T x (Q_J^-1 x I).

    Q_J = tau * graph Laplacian + epsilon * I. Factor is [N,HJC,HJC].
    Use native structured operators for large trajectories, not this dense helper.
    Hyperparameters and graph must be selected without calibration/test outcomes.
    """
    temporal_factor = tensor(temporal_factor, "temporal_factor", 3)
    adjacency = tensor(adjacency, "adjacency", 2).to(temporal_factor)
    tau, epsilon = positive(tau, "tau"), positive(epsilon, "epsilon")
    if not isinstance(coordinates, int) or coordinates < 1:
        raise ValueError("coordinates must be a positive integer")
    if temporal_factor.shape[-2] != temporal_factor.shape[-1]:
        raise ValueError("Temporal factors must be square")
    if (
        adjacency.shape[0] != adjacency.shape[1]
        or (adjacency < 0).any()
        or not torch.allclose(adjacency, adjacency.T)
    ):
        raise ValueError("Adjacency must be symmetric, square and nonnegative")
    j = adjacency.shape[0]
    eye = torch.eye(j, device=adjacency.device, dtype=adjacency.dtype)
    precision = tau * (torch.diag(adjacency.sum(1)) - adjacency) + epsilon * eye
    q_chol = torch.linalg.cholesky(precision)
    joint_cov = torch.cholesky_solve(eye, q_chol)
    joint_factor = torch.kron(
        torch.linalg.cholesky(joint_cov).contiguous(),
        torch.eye(coordinates, device=adjacency.device, dtype=adjacency.dtype),
    )
    return torch.stack(
        [torch.kron(t.contiguous(), joint_factor.contiguous()) for t in temporal_factor]
    )
