"""SIGReg — Sketched Isotropic Gaussian Regularization (Balestriero & LeCun 2025)."""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._dcca import DCCA


def _invariance_loss(representations: list[torch.Tensor]) -> torch.Tensor:
    """Compute the mean pairwise MSE across all views.

    Args:
        representations: List of tensors, each of shape
            (batch_size, latent_dimensions).

    Returns:
        Scalar tensor: the mean of ``MSE(z_i, z_j)`` over all unordered
        view pairs.
    """
    pairs = itertools.combinations(range(len(representations)), 2)
    losses = [
        F.mse_loss(representations[i], representations[j]) for i, j in pairs
    ]
    return torch.stack(losses).mean()


def _sigreg_directions(
    latent_dimensions: int,
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample random unit directions on the sphere S^{K-1}.

    Args:
        latent_dimensions: Ambient dimensionality K.
        num_directions: Number of directions M to sample.
        device: Device to place the directions on.
        dtype: Floating point dtype for the directions.

    Returns:
        Tensor of shape (K, M) with unit-norm columns.
    """
    directions = torch.randn(
        latent_dimensions, num_directions, device=device, dtype=dtype
    )
    unit_directions: torch.Tensor = directions / directions.norm(dim=0, keepdim=True)
    return unit_directions


def _sigreg_loss(
    z: torch.Tensor,
    directions: torch.Tensor,
    quad_nodes: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the SIGReg (Epps-Pulley) statistic for one view's batch.

    Projects the batch onto each direction and tests the resulting
    one-dimensional empirical characteristic function against that of a
    standard normal, integrated against the Gaussian weight
    $e^{-t^2/2}$ via fixed Gauss-Hermite quadrature.

    Args:
        z: Batch of embeddings, shape (n, K).
        directions: Unit directions, shape (K, M).
        quad_nodes: Quadrature nodes $t_k$, shape (Q,).
        quad_weights: Quadrature weights for $\int \cdot\, e^{-t^2/2}\,dt$,
            shape (Q,).

    Returns:
        Scalar tensor: the statistic averaged over the M directions.
    """
    projections = z @ directions  # (n, M)
    angles = quad_nodes.view(-1, 1, 1) * projections.unsqueeze(0)  # (Q, n, M)
    cos_mean = angles.cos().mean(dim=1)  # (Q, M)
    sin_mean = angles.sin().mean(dim=1)  # (Q, M)
    target = torch.exp(-0.5 * quad_nodes**2).view(-1, 1)  # (Q, 1)
    squared_deviation = (cos_mean - target).pow(2) + sin_mean.pow(2)  # (Q, M)
    per_direction = (quad_weights.view(-1, 1) * squared_deviation).sum(dim=0)  # (M,)
    return per_direction.mean()


class SIGReg(DCCA):
    r"""Sketched Isotropic Gaussian Regularization for self-supervised CCA.

    Trains encoders under an invariance loss between views' embeddings,
    regularised so that each view's embedding distribution matches an
    isotropic Gaussian $\mathcal{N}(0, I_K)$ — the distribution shown by
    Balestriero and LeCun to minimise worst-case downstream probing risk,
    from which any collapsed solution (zero variance along some direction)
    is maximally distant. By the Cramér-Wold theorem this
    high-dimensional constraint reduces to univariate goodness-of-fit
    tests along random one-dimensional projections, evaluated here via
    the Epps-Pulley characteristic-function statistic:

    $$
    \mathcal{L} = \mathcal{L}_{\text{inv}} + \lambda \mathcal{L}_{\text{SIGReg}},
    \qquad
    \mathcal{L}_{\text{inv}} = \binom{M}{2}^{-1}\!\!\sum_{i<j}
        \operatorname{MSE}(z_i, z_j)
    $$

    $$
    \mathcal{L}_{\text{SIGReg}} = \frac{1}{V}\sum_{v=1}^{V} \frac{1}{M}
        \sum_{m=1}^{M} \int \left|
            \frac{1}{n}\sum_{i=1}^n e^{\mathrm{i}t\langle z_{v,i}, a_m\rangle}
            - e^{-t^2/2}
        \right|^2 e^{-t^2/2}\, dt
    $$

    where $a_1, \dots, a_M$ are directions resampled uniformly on
    $\mathbb{S}^{K-1}$ at every call, $V$ is the number of views and
    the $t$-integral is evaluated by fixed Gauss-Hermite quadrature. Unlike
    :class:`~cca_zoo.deep.VICReg`'s second-moment (variance + covariance)
    penalty, SIGReg constrains the full marginal shape of the embedding and
    provably excludes the collapsed solution, at the cost of a single
    hyperparameter $\lambda$ instead of three. With linear encoders this
    recovers a stochastic, whitening-free linear CCA-type subspace, in the
    same family as :class:`~cca_zoo.linear.gradient.CCA_EY`.

    References:
        Balestriero, R., & LeCun, Y. "LeJEPA: Provable and Scalable
        Self-Supervised Learning without the Heuristics." arXiv preprint
        arXiv:2511.08544 (2025).

        Kuhn, L., Serra, G., Balestriero, R., & Buettner, F. "LeVJEPA:
        Efficient & Scalable Video Pretraining without the Heuristics."
        arXiv preprint arXiv:2608.27395 (2026).

    Args:
        latent_dimensions: Dimensionality K of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        sigreg_coeff: Weight $\lambda$ for the SIGReg term. Default is 0.02,
            following Balestriero and LeCun's fixed setting.
        num_directions: Number of random directions M sampled per
            evaluation for the Cramér-Wold projection. Default is 16.
        num_quad_points: Number of Gauss-Hermite quadrature nodes Q used to
            approximate the $t$-integral. Default is 17.
        objective: Ignored; the SIGReg loss is fixed. Accepted for API
            compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Unused. Present for API compatibility. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = SIGReg(latent_dimensions=4, encoders=[enc1, enc2])
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        sigreg_coeff: float = 0.02,
        num_directions: int = 16,
        num_quad_points: int = 17,
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            objective=objective,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.sigreg_coeff = sigreg_coeff
        self.num_directions = num_directions

        nodes, weights = np.polynomial.hermite.hermgauss(num_quad_points)
        # Change of variables t = sqrt(2) s turns the Gauss-Hermite rule for
        # weight e^{-s^2} into a rule for e^{-t^2/2}: see class docstring.
        self.register_buffer(
            "quad_nodes", torch.as_tensor(np.sqrt(2) * nodes, dtype=torch.float32)
        )
        self.register_buffer(
            "quad_weights", torch.as_tensor(np.sqrt(2) * weights, dtype=torch.float32)
        )
        self.quad_nodes: torch.Tensor
        self.quad_weights: torch.Tensor

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the SIGReg objective.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions), one per view.
            independent_representations: Unused.

        Returns:
            Dictionary with keys ``"objective"``, ``"invariance"``, and
            ``"sigreg"``.
        """
        z0 = representations[0]
        directions = _sigreg_directions(
            self.latent_dimensions, self.num_directions, z0.device, z0.dtype
        )
        quad_nodes = self.quad_nodes.to(dtype=z0.dtype)
        quad_weights = self.quad_weights.to(dtype=z0.dtype)

        invariance = _invariance_loss(representations)
        sigreg = torch.stack(
            [
                _sigreg_loss(z, directions, quad_nodes, quad_weights)
                for z in representations
            ]
        ).mean()
        objective = invariance + self.sigreg_coeff * sigreg
        return {
            "objective": objective,
            "invariance": invariance,
            "sigreg": sigreg,
        }
