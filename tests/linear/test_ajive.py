"""AJIVE: joint and individual structure, checked against planted truth."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from cca_zoo.linear import AJIVE, CCA, GCCA
from tests._helpers import principal_cosines


def _planted(
    seed: int = 0,
    n: int = 150,
    widths: tuple[int, ...] = (40, 60),
    n_joint: int = 2,
    n_individual: tuple[int, ...] = (1, 2),
    noise: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Views of joint and per-view individual factors, with the factors."""
    rng = np.random.default_rng(seed)
    joint = rng.standard_normal((n, n_joint))
    individual, views = [], []
    for p, r in zip(widths, n_individual):
        own = rng.standard_normal((n, r))
        individual.append(own)
        views.append(
            3 * joint @ rng.standard_normal((n_joint, p)) / np.sqrt(p)
            + 3 * own @ rng.standard_normal((r, p)) / np.sqrt(p)
            + noise * rng.standard_normal((n, p))
        )
    return views, joint, individual


@pytest.mark.parametrize("n_components", [2, "auto"])
def test_joint_and_individual_structure_is_recovered(n_components: int | str) -> None:
    """Joint and individual scores span the planted factors, in the right ranks."""
    views, joint, individual = _planted()
    model = AJIVE(n_components=n_components, random_state=0).fit(views)
    assert model.n_components_ == 2
    assert model.signal_ranks_ == [3, 4]
    assert [s.shape[1] for s in model.individual_scores_] == [1, 2]
    assert principal_cosines(model.joint_scores_, joint).min() > 0.97
    for scores, truth in zip(model.individual_scores_, individual):
        assert principal_cosines(scores, truth).min() > 0.9
        # Individual scores are orthogonal to the joint ones in sample space.
        cross = model.joint_scores_.T @ scores / (len(scores) - 1)
        np.testing.assert_allclose(cross, 0, atol=1e-10)


def test_scores_are_unit_variance_and_uncorrelated() -> None:
    """Joint scores, and each view's individual scores, have identity covariance."""
    views, _, _ = _planted()
    model = AJIVE(n_components=2).fit(views)
    for scores in [model.joint_scores_, *model.individual_scores_]:
        np.testing.assert_allclose(
            np.cov(scores.T), np.eye(scores.shape[1]), atol=1e-10
        )


def test_variance_is_split_into_joint_individual_and_noise() -> None:
    """Each view's variance is the joint, individual and noise parts, which add up."""
    views, _, _ = _planted()
    model = AJIVE(n_components=2).fit(views)
    for i, view in enumerate(views):
        centred = view - view.mean(axis=0)
        joint = model.joint_scores_ @ model.joint_loadings_[i].T
        individual = model.individual_scores_[i] @ model.individual_loadings_[i].T
        total = np.sum(centred**2)
        np.testing.assert_allclose(
            model.joint_variance_explained_[i],
            np.sum(
                (model.joint_scores_[:, :, None] * model.joint_loadings_[i].T[None])
                ** 2,
                axis=(0, 2),
            )
            / total,
        )
        assert np.sum(joint**2) / total == pytest.approx(
            model.joint_variance_explained_[i].sum()
        )
        assert np.sum(individual**2) / total == pytest.approx(
            model.individual_variance_explained_[i].sum()
        )
        assert np.sum((centred - joint - individual) ** 2) / total == pytest.approx(
            model.noise_variance_explained_[i]
        )
        assert (
            model.joint_variance_explained_[i].sum()
            + model.individual_variance_explained_[i].sum()
            + model.noise_variance_explained_[i]
        ) == pytest.approx(1.0)


def test_a_strong_private_factor_does_not_leak_into_the_joint_scores() -> None:
    """Joint scores for new samples follow the joint factor, not each view's own."""
    views, joint, _ = _planted(n=400, n_individual=(2, 2), noise=0.3)
    train, test = [v[:200] for v in views], [v[200:] for v in views]
    model = AJIVE(n_components=2).fit(train)
    for scores in model.transform(test):
        assert principal_cosines(scores, joint[200:]).min() > 0.95
    # The individual scores of new samples agree with the training ones.
    for fitted, scores in zip(
        model.individual_scores_, model.transform_individual(train)
    ):
        assert principal_cosines(fitted, scores).min() > 0.99


def test_auto_finds_no_joint_structure_in_independent_views() -> None:
    """With nothing shared, "auto" raises rather than invent joint structure."""
    rng = np.random.default_rng(0)
    views = [
        3 * rng.standard_normal((150, 3)) @ rng.standard_normal((3, p))
        + rng.standard_normal((150, p))
        for p in (40, 60)
    ]
    with pytest.raises(ValueError, match="no joint structure was found"):
        AJIVE(n_components="auto", random_state=0).fit(views)


def test_alpha_sets_the_random_direction_bound() -> None:
    """A higher level keeps fewer chance alignments: here, a lower bound keeps one."""
    rng = np.random.default_rng(0)
    views = [
        3 * rng.standard_normal((150, 3)) @ rng.standard_normal((3, p))
        + rng.standard_normal((150, p))
        for p in (40, 60)
    ]
    strict = AJIVE(n_components="auto", alpha=0.05, random_state=0)
    lax = AJIVE(n_components="auto", alpha=0.99, random_state=0)
    with pytest.raises(ValueError, match="no joint structure was found"):
        strict.fit(views)
    assert lax.fit(views).n_components_ >= 1


def test_ranks_can_be_set() -> None:
    """Signal and individual ranks given, by view or for all, are used."""
    views, _, _ = _planted()
    model = AJIVE(n_components=1, signal_ranks=[5, 6], individual_ranks=[0, 3]).fit(
        views
    )
    assert model.signal_ranks_ == [5, 6]
    assert [s.shape[1] for s in model.individual_scores_] == [0, 3]
    assert model.individual_variance_explained_[0].size == 0
    assert AJIVE(signal_ranks=4).fit(views).signal_ranks_ == [4, 4]
    # A signal subspace is at least as large as the joint rank.
    assert AJIVE(n_components=3, signal_ranks=1).fit(views).signal_ranks_ == [3, 3]


def test_more_than_two_views() -> None:
    """The joint space is found across three views."""
    views, joint, _ = _planted(widths=(30, 40, 50), n_individual=(1, 1, 2))
    model = AJIVE(n_components="auto", random_state=0).fit(views)
    assert model.n_components_ == 2
    assert principal_cosines(model.joint_scores_, joint).min() > 0.95


def test_joint_rank_is_at_most_what_the_data_allow() -> None:
    """A joint rank above the narrowest view's features raises."""
    views, _, _ = _planted(n=20, widths=(5, 4))
    with pytest.raises(ValueError, match="n_components=5 must be at most 4"):
        AJIVE(n_components=5).fit(views)


def test_two_views_the_joint_scores_are_cca_on_the_signal_subspaces() -> None:
    """With the views cut to their signal subspaces, the joint scores are CCA's.

    Each joint score is the normalised sum of the two canonical variates of the
    PCA-reduced views.
    """
    views, _, _ = _planted(widths=(40, 60))
    rank = 4
    model = AJIVE(n_components=2, signal_ranks=rank, individual_ranks=0).fit(views)
    reduced = [PCA(rank).fit_transform(v) for v in views]
    z1, z2 = CCA(n_components=2).fit(reduced).transform(reduced)
    for k in range(2):
        corr = np.corrcoef(model.joint_scores_[:, k], z1[:, k] + z2[:, k])[0, 1]
        assert abs(corr) == pytest.approx(1.0, abs=1e-8)


def test_more_views_the_joint_space_is_gcca_on_the_signal_subspaces() -> None:
    """For more views the joint space is multiset CCA's on the PCA-reduced views."""
    views, _, _ = _planted(widths=(30, 40, 50), n_individual=(1, 1, 2))
    rank = 4
    model = AJIVE(n_components=2, signal_ranks=rank, individual_ranks=0).fit(views)
    reduced = [PCA(rank).fit_transform(v) for v in views]
    latent = np.mean(GCCA(n_components=2).fit(reduced).transform(reduced), axis=0)
    np.testing.assert_allclose(
        principal_cosines(model.joint_scores_, latent), 1.0, atol=1e-8
    )
