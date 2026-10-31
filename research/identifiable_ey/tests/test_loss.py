import numpy as np
import pytest
import torch

from ieye.loss import (
    SecondMoments,
    cf_distance,
    cf_distance_population_laplace,
    gauss_hermite,
    identifiable_ey_loss,
    shared_std,
)

NODES, WEIGHTS = gauss_hermite(17)


def laplace(shape, generator):
    u = torch.rand(shape, generator=generator, dtype=torch.float64) - 0.5
    return -torch.sign(u) * torch.log1p(-2 * u.abs()) / np.sqrt(2)


def test_cf_distance_is_unbiased():
    g = torch.Generator().manual_seed(0)
    n, reps = 6, 40000
    z = laplace((reps, n, 1), g)
    estimate = torch.stack([cf_distance(zi, NODES, WEIGHTS) for zi in z]).mean()
    exact = cf_distance_population_laplace(NODES, WEIGHTS)
    se = torch.stack([cf_distance(zi, NODES, WEIGHTS) for zi in z]).std() / reps**0.5
    assert abs(estimate - exact) < 4 * se


def test_gaussian_has_zero_distance_in_expectation():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(20000, 8, 1, generator=g, dtype=torch.float64)
    values = torch.stack([cf_distance(zi, NODES, WEIGHTS) for zi in z])
    assert abs(values.mean()) < 4 * values.std() / 20000**0.5


def test_ey_penalty_with_independent_batch_is_unbiased():
    from ieye.loss import ey_penalty_and_reward

    g = torch.Generator().manual_seed(2)
    mix = torch.randn(3, 3, generator=g, dtype=torch.float64)

    def draw(n):
        s = torch.randn(n, 3, generator=g, dtype=torch.float64)
        return [s @ mix, s @ mix + 0.5 * torch.randn(n, 3, generator=g, dtype=torch.float64)]

    big = draw(400000)
    _, truth = ey_penalty_and_reward(big, big)
    n, reps = 10, 20000
    two_batch = torch.stack([ey_penalty_and_reward(draw(n), draw(n))[1] for _ in range(reps)])
    one_batch = torch.stack([ey_penalty_and_reward(b := draw(n), b)[1] for _ in range(reps)])
    se = two_batch.std() / reps**0.5
    assert abs(two_batch.mean() - truth) < 4 * se
    assert abs(one_batch.mean() - truth) > 10 * se  # the single-batch form is biased


@pytest.mark.parametrize("variate", ["average", "views"])
def test_loss_gradcheck(variate):
    g = torch.Generator().manual_seed(3)
    n, p, k = 30, 4, 2
    batch = [torch.randn(n, p, generator=g, dtype=torch.float64) for _ in range(2)]
    independent = [torch.randn(n, p, generator=g, dtype=torch.float64) for _ in range(2)]
    moments = SecondMoments(torch.randn(p, p, generator=g, dtype=torch.float64) / 4)
    w0 = [torch.randn(p, k, generator=g, dtype=torch.float64) for _ in range(2)]

    def f(w1, w2):
        return identifiable_ey_loss(
            batch, [w1, w2], moments, mu=0.7, variate=variate,
            nodes=NODES, weights=WEIGHTS, independent=independent,
        )["objective"]

    assert torch.autograd.gradcheck(f, [e.clone().requires_grad_() for e in w0])


def test_ng_is_scale_invariant():
    g = torch.Generator().manual_seed(4)
    batch = [torch.randn(50, 3, generator=g, dtype=torch.float64) for _ in range(2)]
    moments = SecondMoments(torch.randn(3, 3, generator=g, dtype=torch.float64) / 4)
    w = [torch.randn(3, 2, generator=g, dtype=torch.float64) for _ in range(2)]
    kwargs = dict(mu=1.0, variate="average", nodes=NODES, weights=WEIGHTS)
    a = identifiable_ey_loss(batch, w, moments, **kwargs)["ng"]
    b = identifiable_ey_loss(batch, [3 * e for e in w], moments, **kwargs)["ng"]
    assert torch.allclose(a, b)
    assert shared_std(w, moments, "average").shape == (2,)


def test_denoised_ng_is_unbiased_for_the_signal_and_gradchecks():
    from ieye.loss import legendre

    nodes, weights = legendre(24, 2.0)
    g = torch.Generator().manual_seed(5)
    # u = alpha * s + noise, s unit-variance Laplace: the denoised distance is the
    # Laplace signal's distance (scaled by alpha), however much noise is added.
    alpha2 = torch.tensor([0.4], dtype=torch.float64)
    n, reps = 8, 20000
    s = laplace((reps, n, 1), g)
    u = alpha2.sqrt() * s + (1 - alpha2).sqrt() * torch.randn(reps, n, 1, generator=g, dtype=torch.float64)
    values = torch.stack([cf_distance(ui, nodes, weights, 1 - alpha2) for ui in u])
    phi = 1 / (1 + alpha2 * nodes**2 / 2)  # CF of alpha * s
    gaussian = torch.exp(-(nodes**2) / 2)
    exact = ((phi - gaussian * torch.exp((1 - alpha2) * nodes**2 / 2)) ** 2) @ weights
    assert abs(values.mean() - exact) < 4 * values.std() / reps**0.5

    n, p, k = 30, 4, 2
    batch = [torch.randn(n, p, generator=g, dtype=torch.float64) for _ in range(2)]
    shared = torch.randn(p, k, generator=g, dtype=torch.float64)
    moments = SecondMoments(torch.eye(p, dtype=torch.float64) * 0.5)
    w0 = [shared + 0.1 * torch.randn(p, k, generator=g, dtype=torch.float64) for _ in range(2)]

    def f(w1, w2):
        return identifiable_ey_loss(
            batch, [w1, w2], moments, mu=1.0, variate="average",
            nodes=nodes, weights=weights, denoise=True,
        )["objective"]

    assert torch.autograd.gradcheck(f, [e.clone().requires_grad_() for e in w0])


def test_u_stat_sigreg_is_unbiased_under_the_gaussian_null():
    from cca_zoo.deep._lejepa import _sigreg

    from ieye.ssl import N_POINTS, N_SLICES, T_MAX, sigreg_u

    g = torch.Generator().manual_seed(6)
    n, reps = 16, 400
    zs = [torch.randn(n, 8, generator=g) for _ in range(reps)]
    u = torch.stack([sigreg_u(z, i) for i, z in enumerate(zs)])
    v = torch.stack([_sigreg(z, i, N_SLICES, N_POINTS, T_MAX) for i, z in enumerate(zs)])
    assert abs(u.mean()) < 4 * u.std() / reps**0.5
    assert v.mean() > 10 * v.std() / reps**0.5  # the V-statistic is biased upwards
