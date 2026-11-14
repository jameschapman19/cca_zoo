"""LeJEPA: a joint-embedding predictive architecture regularised by SIGReg."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch


def _sigreg(
    z: torch.Tensor,
    seed: int,
    n_slices: int,
    n_points: int,
    t_max: float,
    unbiased: bool = False,
) -> torch.Tensor:
    """SIGReg: the Epps-Pulley statistic of random 1-D projections of ``z``.

    Each projection's empirical characteristic function is compared with the
    standard normal's, weighted by it, on ``n_points`` trapezoidal points of
    ``[-t_max, t_max]``, and scaled by the batch size. The ``n_slices`` unit
    directions come from a generator seeded with ``seed``, so every view
    shares them within a step and they are resampled across steps.

    The plug-in statistic is biased: for an exactly standard normal ``z`` its
    expectation, after the factor of the batch size, is close to 1 rather than
    0, because the squared modulus of the empirical characteristic function
    includes each sample's product with itself. With ``unbiased``, those
    ``i = j`` terms are dropped, so the expectation is 0 for a standard normal.
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
    # |ecf - phi|^2 = |ecf|^2 - 2 Re(ecf) phi + phi^2, with ecf = E[cos] + i E[sin]
    # and phi real.
    cos_mean, sin_mean = xt.cos().mean(0), xt.sin().mean(0)
    modulus_sq = cos_mean.square() + sin_mean.square()
    if unbiased:
        # |ecf|^2 sums over all pairs (i, j), including the n terms with i = j,
        # each equal to 1; dropping them leaves an unbiased estimate of |phi_z|^2.
        n = z.shape[0]
        modulus_sq = (n * modulus_sq - 1) / (n - 1)
    error = modulus_sq - 2 * cos_mean * gaussian_cf + gaussian_cf.square()
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

    SIGReg leaves the embedding's rotation free, so the axes are not named. With
    ``ordered=True`` the squared distance of coordinate $j$ to the centre is
    weighted by $w_j$, falling geometrically from $\sqrt{r}$ to $1/\sqrt{r}$
    (``order_ratio`` $r$) and scaled to average 1. The loss is then lowest when
    the coordinates are the canonical variates of the views, ordered by
    cross-view correlation. SIGReg also controls the embedding's distribution.
    In a test world of Gaussian factors of unequal stability, that kept functions
    of the most stable factor from taking the coordinates that a less stable
    factor needed, as they do under :class:`~cca_zoo.deep.DCCAEY`; see
    :func:`~cca_zoo.model_selection.nonlinear_redundancy` to check an embedding.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view; two or more.
        lam: Weight $\lambda \in [0, 1]$ of SIGReg against the predictive
            term. Default is 0.05.
        n_slices: Random directions per step. Default is 256.
        n_points: Quadrature points of the Epps-Pulley integral. Default is 17.
        t_max: Half-width of the integration interval. Default is 5.0.
        learning_rate: Adam learning rate. Default is 1e-3.
        ordered: Whether to weight the alignment of coordinate j by a weight
            falling with j, so that the coordinates are ordered canonical
            variates of the views instead of an arbitrary rotation of them.
            Default is False.
        order_ratio: Ratio of the first coordinate's weight to the last's when
            ``ordered``, at least 1; 1 weights all coordinates equally. The
            last coordinate's pull towards alignment is ``1 / order_ratio`` of
            the first's, so a large ratio, like a large ``lam``, leaves the
            trailing coordinates to SIGReg. With linear encoders and two
            coordinates the canonical variates were recovered for ratios 2 to
            8 at ``lam`` up to 0.05, and not at ratio 16 with ``lam`` 0.05 or
            at ``lam`` 0.2. Default is 4.
        unbiased: Whether SIGReg uses the unbiased estimate of the squared
            characteristic-function error (see :func:`_sigreg`) in place of the
            plug-in one, which is biased upward by about 1 for a standard-normal
            batch. Default is False, the reference implementation's.

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
        ordered: bool = False,
        order_ratio: float = 4.0,
        unbiased: bool = False,
    ) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}.")
        if order_ratio < 1.0:
            raise ValueError(f"order_ratio must be at least 1, got {order_ratio}.")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.lam = lam
        self.n_slices = n_slices
        self.n_points = n_points
        self.t_max = t_max
        self.ordered = ordered
        self.order_ratio = order_ratio
        self.unbiased = unbiased

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The LeJEPA loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "sim_loss", "sigreg"}``.
        """
        representations = self(batch["views"])
        z = torch.stack(representations)  # (views, n, d)
        k = self.n_components
        w = np.geomspace(self.order_ratio**0.5, self.order_ratio**-0.5, k)
        weights = torch.as_tensor(
            w / w.mean() if self.ordered else np.ones(k), dtype=z.dtype, device=z.device
        )
        sim = ((z.mean(dim=0) - z).square().mean(dim=(0, 1)) * weights).mean()
        sigreg = torch.stack(
            [
                _sigreg(
                    zk,
                    self.global_step,
                    self.n_slices,
                    self.n_points,
                    self.t_max,
                    self.unbiased,
                )
                for zk in representations
            ]
        ).mean()
        return {
            "objective": (1 - self.lam) * sim + self.lam * sigreg,
            "sim_loss": sim,
            "sigreg": sigreg,
        }
