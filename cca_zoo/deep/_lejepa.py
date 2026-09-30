"""LeJEPA: a joint-embedding predictive architecture regularised by SIGReg."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch


def _sigreg(
    z: torch.Tensor, seed: int, n_slices: int, n_points: int, t_max: float
) -> torch.Tensor:
    """SIGReg: the Epps-Pulley statistic of random 1-D projections of ``z``.

    Each projection's empirical characteristic function is compared with the
    standard normal's, weighted by it, on ``n_points`` trapezoidal points of
    ``[-t_max, t_max]``, and scaled by the batch size. The ``n_slices`` unit
    directions come from a generator seeded with ``seed``, so every view
    shares them within a step and they are resampled across steps.
    """
    generator = torch.Generator(device=z.device)
    generator.manual_seed(seed)
    directions = torch.randn(
        z.shape[1], n_slices, generator=generator, device=z.device, dtype=z.dtype
    )
    directions = directions / directions.norm(dim=0)
    t = torch.linspace(-t_max, t_max, n_points, device=z.device, dtype=z.dtype)
    gaussian_cf = torch.exp(-0.5 * t**2)
    xt = (z @ directions).unsqueeze(2) * t  # (n, slices, points)
    # |ecf - phi|^2, with ecf = E[cos] + i E[sin] and phi real.
    error = (xt.cos().mean(0) - gaussian_cf).square() + xt.sin().mean(0).square()
    return (torch.trapezoid(error * gaussian_cf, t, dim=1) * z.shape[0]).mean()


class LeJEPA(BaseDeep):
    r"""LeJEPA: every view predicts the views' centre, and SIGReg prevents collapse.

    $$
    \mathcal{L} = (1 - \lambda)\,\frac{1}{K n d} \sum_{k=1}^{K}
        \lVert z_k - \bar z \rVert_F^2
        + \lambda\,\frac{1}{K} \sum_{k=1}^{K} \operatorname{SIGReg}(z_k),
    \qquad \bar z = \frac{1}{K} \sum_{k=1}^{K} z_k,
    $$

    where SIGReg is the Epps-Pulley test statistic of $z_k$'s projections on
    ``n_slices`` random unit directions against a standard normal, averaged
    over directions, which are resampled at every training step. Isotropic
    Gaussian embeddings, which SIGReg favours, cannot collapse, so no
    stop-gradient, predictor or teacher network is needed. Every view is a
    global view, so the centre is the mean over all views.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view; two or more.
        lam: Weight $\lambda \in [0, 1]$ of SIGReg against the predictive
            term. Default is 0.05.
        n_slices: Random directions per step. Default is 256.
        n_points: Quadrature points of the Epps-Pulley integral. Default is 17.
        t_max: Half-width of the integration interval. Default is 5.0.
        learning_rate: Adam learning rate. Default is 1e-3.

    Raises:
        ValueError: If ``lam`` is outside ``[0, 1]``.

    References:
        Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable
        Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import LeJEPA
        >>> model = LeJEPA(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        lam: float = 0.05,
        n_slices: int = 256,
        n_points: int = 17,
        t_max: float = 5.0,
        learning_rate: float = 1e-3,
    ) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.lam = lam
        self.n_slices = n_slices
        self.n_points = n_points
        self.t_max = t_max

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The LeJEPA loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "sim_loss", "sigreg"}``.
        """
        representations = self(batch["views"])
        z = torch.stack(representations)  # (views, n, d)
        sim = (z.mean(dim=0) - z).square().mean()
        sigreg = torch.stack(
            [
                _sigreg(zk, self.global_step, self.n_slices, self.n_points, self.t_max)
                for zk in representations
            ]
        ).mean()
        return {
            "objective": (1 - self.lam) * sim + self.lam * sigreg,
            "sim_loss": sim,
            "sigreg": sigreg,
        }
