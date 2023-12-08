from typing import List

import torch


def torch_cross_cov(A, B):
    A = A.T
    B = B.T

    A = A - A.mean(dim=1, keepdim=True)
    B = B - B.mean(dim=1, keepdim=True)

    C = A @ B.T
    return C / (A.size(1) - 1)


@torch.jit.script
def inv_sqrtm(A: torch.Tensor, eps: float = 1e-9):
    """Compute the inverse square-root of a positive definite matrix."""
    # Perform eigendecomposition of covariance matrix
    U, S, V = torch.svd(A)
    # Enforce positive definite by taking a torch max() with eps
    S = torch.clamp(S, min=eps)
    # S = torch.max(S, torch.tensor(eps, device=S.device))
    # Calculate inverse square-root
    inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
    # Calculate inverse square-root matrix
    B = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))
    return B


def CCA_AB(representations: List[torch.Tensor]):
    latent_dimensions = representations[0].shape[1]
    A = torch.zeros(
        latent_dimensions, latent_dimensions, device=representations[0].device
    )  # initialize the cross-covariance matrix
    B = torch.zeros(
        latent_dimensions, latent_dimensions, device=representations[0].device
    )  # initialize the auto-covariance matrix
    for i, zi in enumerate(representations):
        B.add_(torch.cov(zi.T))  # In-place addition
        for j, zj in enumerate(representations):
            A.add_(torch_cross_cov(zi, zj))  # In-place addition

    A.div_(len(representations))  # In-place division
    B.div_(len(representations))  # In-place division
    return A, B
