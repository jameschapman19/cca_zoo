from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ._dcca import DCCA


class BatchWhiten(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(BatchWhiten, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if self.track_running_stats:
            self.register_buffer(
                "running_covar", torch.eye(num_features, **factory_kwargs)
            )
            self.running_covar: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_covar", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            # fill with identity to preserve initialization
            self.running_covar.fill_diagonal_(1)
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self):
        self.reset_running_stats()

    def forward(self, input: Tensor) -> Tensor:
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_covar is None

        running_covar = (
            self.running_covar
            if not self.training or self.track_running_stats
            else None
        )

        # Calculate batch covariance
        covar = torch.matmul(input.T, input) / input.shape[0]

        # Update running covariance
        if bn_training:
            with torch.no_grad():
                if running_covar is not None:
                    running_covar.mul_(exponential_average_factor).add_(
                        covar, alpha=1 - exponential_average_factor
                    )

                # Calculate whitened input
                if running_covar is not None:
                    covar = running_covar
                # Enforce positive definite by taking a torch max() with eps
                covar = torch.max(covar, torch.tensor(self.eps, device=covar.device))
                # Calculate inverse square-root matrix
                B = inv_sqrtm(covar, self.eps)
                # Calculate whitened input
                input = torch.matmul(input, B)
                return input
        else:
            return input


def inv_sqrtm(A, eps=1e-9):
    """Compute the inverse square-root of a positive definite matrix."""
    # Perform eigendecomposition of covariance matrix
    U, S, V = torch.svd(A)
    # Enforce positive definite by taking a torch max() with eps
    S = torch.max(S, torch.tensor(eps, device=S.device))
    # Calculate inverse square-root
    inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
    # Calculate inverse square-root matrix
    B = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))
    return B


class DCCA_NOI(DCCA):
    """
    A class used to fit a DCCA model by non-linear orthogonal iterations


    References
    ----------
    Wang, Weiran, et al. "Stochastic optimization for deep CCA via nonlinear orthogonal iterations." 2015 53rd Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2015.

    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        r: float = 0,
        rho: float = 0.1,
        eps: float = 1e-9,
        **kwargs,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            r=r,
            eps=eps,
            **kwargs,
        )
        if rho < 0 or rho > 1:
            raise ValueError(f"rho should be between 0 and 1. rho={rho}")
        self.eps = eps
        self.rho = rho
        self.mse = torch.nn.MSELoss(reduction="sum")
        # Replace BatchNorm1d with BatchWhiten
        self.bws = torch.nn.ModuleList(
            [BatchWhiten(latent_dimensions, momentum=rho) for _ in self.encoders]
        )

    def loss(self, batch, **kwargs):
        z = self(batch["views"])
        z_w = [bw(z_) for z_, bw in zip(z, self.bws)]
        loss = self.mse(z[0], z_w[1].detach()) + self.mse(z[1], z_w[0].detach())
        return {"objective": loss}
