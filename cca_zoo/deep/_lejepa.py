"""LeJEPA — Joint-Embedding Predictive Architecture with SIGReg (Balestriero 2025)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA


def _sigreg(
    z: torch.Tensor, seed: int, num_slices: int, n_points: int, t_max: float
) -> torch.Tensor:
    r"""Sketched Isotropic Gaussian Regularisation via the Epps-Pulley statistic.

    Projects ``z`` onto ``num_slices`` random unit directions (drawn from a
    generator seeded with ``seed`` so every view shares the same slices within
    a step while the slices are resampled across steps), then measures the
    Gaussian-weighted squared distance between each projection's empirical
    characteristic function and that of $\mathcal{N}(0, 1)$.

    Args:
        z: Tensor of shape (batch_size, latent_dimensions).
        seed: Seed for the slice directions, typically the global step.
        num_slices: Number of random directions.
        n_points: Number of quadrature points on $[-t_{max}, t_{max}]$.
        t_max: Half-width of the integration interval.

    Returns:
        Scalar Epps-Pulley statistic averaged over slices.
    """
    g = torch.Generator(device=z.device)
    g.manual_seed(seed)
    A = torch.randn(z.shape[1], num_slices, generator=g, device=z.device, dtype=z.dtype)
    A = A / A.norm(p=2, dim=0)
    t = torch.linspace(-t_max, t_max, n_points, device=z.device, dtype=z.dtype)
    target_cf = torch.exp(-0.5 * t**2)
    x_t = (z @ A).unsqueeze(2) * t  # (N, M, T)
    # |ecf - phi|^2 with ecf = E[cos] + i E[sin] and phi real.
    err = (x_t.cos().mean(0) - target_cf).square() + x_t.sin().mean(0).square()
    return (torch.trapezoid(err * target_cf, t, dim=1) * z.shape[0]).mean()


class LeJEPA(DCCA):
    r"""LeJEPA: Joint-Embedding Predictive Architecture regularised by SIGReg.

    Treats each view as a view of the same underlying sample: every view's
    embedding is pulled towards the across-view centre, while SIGReg pushes
    each view's embedding distribution towards an isotropic Gaussian, which
    prevents collapse without stop-gradients, predictors or teacher networks:

    $$
    \mathcal{L} = (1 - \lambda)\,\frac{1}{Kd}\sum_{k=1}^{K}
        \bigl\|z_k - \bar z\bigr\|_F^2 / n
        + \lambda\,\frac{1}{K}\sum_{k=1}^{K} \operatorname{SIGReg}(z_k),
    \qquad \bar z = \frac{1}{K}\sum_{k=1}^{K} z_k
    $$

    SIGReg projects $z_k$ onto $M$ random unit directions
    $a_1, \dots, a_M$, resampled at every training step, and averages the
    Epps-Pulley statistic of each projection:

    $$
    \operatorname{SIGReg}(z) = \frac{n}{M}\sum_{m=1}^{M}
        \int \bigl|\hat\varphi_{a_m^\top z}(t) - e^{-t^2/2}\bigr|^2
        e^{-t^2/2}\,dt
    $$

    where $\hat\varphi$ is the empirical characteristic function. The
    integral is evaluated by the trapezoidal rule on ``n_points`` points
    over $[-t_{max}, t_{max}]$.

    In the multiview setting every view is a "global" view, so the centre
    is the mean over all views. The empirical characteristic function is
    computed per device; under DDP each rank regularises its own shard.

    References:
        Balestriero, R., & LeCun, Y. "LeJEPA: Provable and Scalable
        Self-Supervised Learning Without the Heuristics."
        arXiv:2511.08544 (2025).

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
            Two or more views are supported.
        lambd: Trade-off $\lambda \in [0, 1]$ between the predictive
            term and SIGReg. Default is 0.05.
        num_slices: Number of random directions $M$ per step.
            Default is 256.
        n_points: Number of quadrature points. Default is 17.
        t_max: Half-width of the integration interval. Default is 5.0.
        objective: Ignored; the LeJEPA loss is fixed. Accepted for API
            compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Unused. Present for API compatibility. Default is 1e-6.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = LeJEPA(latent_dimensions=4, encoders=[enc1, enc2])
        >>> out = model.loss(model([torch.randn(32, 10), torch.randn(32, 8)]))
        >>> sorted(out)
        ['objective', 'sigreg', 'sim_loss']
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        lambd: float = 0.05,
        num_slices: int = 256,
        n_points: int = 17,
        t_max: float = 5.0,
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
        self.lambd = lambd
        self.num_slices = num_slices
        self.n_points = n_points
        self.t_max = t_max

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the LeJEPA loss.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions), one per view.
            independent_representations: Unused.

        Returns:
            Dictionary with keys ``"objective"``, ``"sim_loss"``, and
            ``"sigreg"``.
        """
        z = torch.stack(representations)  # (K, N, d)
        sim = (z.mean(0) - z).square().mean()
        sigreg = torch.stack(
            [
                _sigreg(
                    zk, self.global_step, self.num_slices, self.n_points, self.t_max
                )
                for zk in representations
            ]
        ).mean()
        return {
            "objective": (1 - self.lambd) * sim + self.lambd * sigreg,
            "sim_loss": sim,
            "sigreg": sigreg,
        }
