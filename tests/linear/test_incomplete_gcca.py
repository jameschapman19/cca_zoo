"""IncompleteGCCA: GCCA that uses the views each sample has."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import GCCA, IncompleteGCCA
from tests._helpers import assert_same_scores_as, principal_cosines


def _factor_views(
    n: int, widths: tuple[int, ...], noise: float = 1.0, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Views of two shared factors, and the factors."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    views = [
        z @ rng.standard_normal((2, p)) + noise * rng.standard_normal((n, p))
        for p in widths
    ]
    return views, z


def _drop(views: list[np.ndarray], rows: list[slice]) -> list[np.ndarray]:
    """Views with the given rows of each set to NaN."""
    dropped = [v.copy() for v in views]
    for v, r in zip(dropped, rows):
        v[r] = np.nan
    return dropped


def test_complete_data_is_gcca() -> None:
    """With no view missing, the fit is GCCA's, up to the sign of each component."""
    views, _ = _factor_views(120, (8, 6, 5))
    for shrinkage in (0.0, 0.3):
        ours = IncompleteGCCA(n_components=2, shrinkage=shrinkage).fit(views)
        gcca = GCCA(n_components=2, shrinkage=shrinkage).fit(views)
        assert_same_scores_as(ours.transform(views), gcca.transform(views))


def test_wide_views_need_no_feature_by_feature_matrix() -> None:
    """Views much wider than they are long fit as GCCA's dual form does."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((60, 2))
    views = [
        z @ rng.standard_normal((2, p)) + rng.standard_normal((60, p))
        for p in (400, 300)
    ]
    ours = IncompleteGCCA(n_components=2, shrinkage=0.5).fit(views)
    gcca = GCCA(n_components=2, shrinkage=0.5).fit(views)
    assert_same_scores_as(ours.transform(views), gcca.transform(views))


def test_samples_missing_views_still_inform_the_fit() -> None:
    """Using partial samples recovers the factors better than complete cases.

    Few samples have every view, and the views are wide, so GCCA on the
    complete cases alone is badly overfitted.
    """
    pool, z = _factor_views(840, (30, 30, 30), noise=1.5)
    views, test = [v[:240] for v in pool], [v[240:] for v in pool]
    # 60 samples miss each view and 60 have all three.
    incomplete = _drop(views, [slice(0, 60), slice(60, 120), slice(120, 180)])
    complete_cases = [v[180:] for v in views]

    def recovery(model: GCCA | IncompleteGCCA) -> float:
        scores = model.transform(test)
        return float(np.mean([principal_cosines(s, z[240:]).min() for s in scores]))

    ours = recovery(IncompleteGCCA(n_components=2).fit(incomplete))
    baseline = recovery(GCCA(n_components=2).fit(complete_cases))
    assert ours > baseline + 0.05


def test_a_sample_in_one_view_has_a_shared_score() -> None:
    """The shared score of a sample is the mean over its observed views."""
    views, _ = _factor_views(100, (6, 5, 4))
    incomplete = _drop(views, [slice(0, 20), slice(0, 10), slice(10, 20)])
    model = IncompleteGCCA(n_components=2).fit(incomplete)
    scores = model.transform(incomplete)
    shared = model.transform_shared(incomplete)
    assert np.isfinite(shared).all()
    # Rows 0-9 are in view 2 only, rows 10-19 in view 1 only.
    np.testing.assert_allclose(shared[:10], scores[2][:10])
    np.testing.assert_allclose(shared[10:20], scores[1][10:20])
    np.testing.assert_allclose(shared[20:], np.mean([s[20:] for s in scores], axis=0))
    assert np.isnan(scores[0][:20]).all() and np.isfinite(scores[0][20:]).all()


def test_predict_fills_a_view_from_the_views_a_sample_has() -> None:
    """A view is reconstructed from whichever of the others each sample has."""
    views, _ = _factor_views(200, (6, 5, 4), noise=0.3)
    incomplete = _drop(views, [slice(0, 30), slice(30, 60), slice(0, 0)])
    model = IncompleteGCCA(n_components=2).fit(incomplete)
    predicted = model.predict([incomplete[0], incomplete[1], None])[2]
    assert np.isfinite(predicted).all()
    error = np.mean((predicted - views[2]) ** 2) / np.var(views[2])
    assert error < 0.6


def test_score_correlates_pairs_on_their_shared_samples() -> None:
    """Views that share few samples still score, on those samples."""
    views, _ = _factor_views(100, (6, 5), noise=0.3)
    incomplete = _drop(views, [slice(0, 40), slice(60, 100)])
    model = IncompleteGCCA(n_components=2).fit(incomplete)
    assert model.score(incomplete) > 0.8
    with pytest.raises(ValueError, match="share at least 2 samples"):
        model.score(_drop(views, [slice(0, 50), slice(50, 100)]))


def test_feature_means_use_the_observed_samples() -> None:
    """Centring ignores the samples that lack the view."""
    views, _ = _factor_views(100, (6, 5))
    views[1] = views[1] + 10.0
    model = IncompleteGCCA().fit(_drop(views, [slice(0, 0), slice(0, 50)]))
    np.testing.assert_allclose(model.means_[1], views[1][50:].mean(axis=0))


def test_rows_must_be_wholly_observed_or_missing() -> None:
    """A row with some entries missing raises, naming the rows."""
    views, _ = _factor_views(50, (6, 5))
    views[0][3, 2] = np.nan
    with pytest.raises(ValueError, match="rows \\[3\\] are partly missing"):
        IncompleteGCCA().fit(views)


def test_every_sample_needs_a_view() -> None:
    """A sample missing from every view raises, naming the samples."""
    views, _ = _factor_views(50, (6, 5))
    with pytest.raises(ValueError, match="samples \\[0 1\\] have none"):
        IncompleteGCCA().fit(_drop(views, [slice(0, 2), slice(0, 2)]))


def test_unlinked_views_raise() -> None:
    """Views joined by too few shared samples have scores that are not tied together."""
    views, _ = _factor_views(60, (6, 5, 4))
    # Views 0 and 1 have samples 0-29; view 2 has 28-59, so shares only 2.
    unlinked = _drop(views, [slice(30, 60), slice(30, 60), slice(0, 28)])
    with pytest.raises(ValueError, match="not linked"):
        IncompleteGCCA(n_components=2).fit(unlinked)
    # More shared samples than components link them.
    linked = _drop(views, [slice(30, 60), slice(30, 60), slice(0, 27)])
    IncompleteGCCA(n_components=2).fit(linked)


def _wide_factor_views(
    n: int,
    widths: tuple[int, ...],
    n_shared: int = 2,
    n_private: int = 3,
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Wide views with a weak shared signal, private factors and noise; the factors."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_shared))
    views = []
    for p in widths:
        own = rng.standard_normal((n, n_private))
        views.append(
            z @ rng.standard_normal((n_shared, p)) * 0.15
            + own @ rng.standard_normal((n_private, p)) * 0.9
            + rng.standard_normal((n, p)) * 0.4
        )
    return views, z


def test_view_ranks_is_gcca_on_principal_components() -> None:
    """Reducing each view to r directions is GCCA on its r principal scores."""
    views, _ = _factor_views(100, (30, 25, 20), noise=0.5)
    ranks = [6, 5, 4]
    reduced = []
    for v, r in zip(views, ranks):
        centred = v - v.mean(axis=0)
        reduced.append(centred @ np.linalg.svd(centred, full_matrices=False)[2][:r].T)
    ours = IncompleteGCCA(n_components=2, view_ranks=ranks).fit(views)
    reference = IncompleteGCCA(n_components=2).fit(reduced)
    assert ours.view_ranks_ == ranks
    assert_same_scores_as(ours.transform(views), reference.transform(reduced))


def test_view_ranks_none_keeps_every_direction() -> None:
    """The default is unchanged by the option existing."""
    views, _ = _factor_views(80, (10, 8))
    assert IncompleteGCCA(n_components=2).fit(views).view_ranks_ is None


def test_auto_view_ranks_find_the_signal_rank() -> None:
    """Views of five strong factors in noise keep about five directions."""
    rng = np.random.default_rng(1)
    views = [
        rng.standard_normal((200, 5)) * 3 @ rng.standard_normal((5, p))
        + rng.standard_normal((200, p))
        for p in (120, 150)
    ]
    model = IncompleteGCCA(n_components=2, view_ranks="auto").fit(views)
    assert model.view_ranks_ == [5, 5]


def test_auto_view_ranks_use_each_views_observed_rows() -> None:
    """A view's rank is read from the samples that have it, not from filled-in rows."""
    views, _ = _wide_factor_views(150, (200, 250, 150))
    views = _drop(views, [slice(100, None), slice(0, 0), slice(0, 0)])
    model = IncompleteGCCA(n_components=2, view_ranks="auto").fit(views)
    assert all(r >= 2 for r in model.view_ranks_)


def test_view_ranks_below_n_components_raise() -> None:
    """A view cannot carry more components than directions it keeps."""
    views, _ = _factor_views(60, (10, 8))
    with pytest.raises(ValueError, match="view_ranks must be at least"):
        IncompleteGCCA(n_components=3, view_ranks=2).fit(views)


def test_reduced_gcca_recovers_a_faint_shared_signal_with_views_missing() -> None:
    """The rank cut regularises: a faint signal is found with 30% of views missing."""
    views, z = _wide_factor_views(150, (300, 400, 200))
    rng = np.random.default_rng(3)
    missing = rng.random((150, 3)) < 0.3
    missing[missing.all(axis=1), 0] = False
    for v, m in zip(views, missing.T):
        v[m] = np.nan
    k = IncompleteGCCA.select_n_components(views, random_state=0)
    assert k == 2
    model = IncompleteGCCA(n_components=k, view_ranks="auto").fit(views)
    assert principal_cosines(model.transform_shared(views), z).min() > 0.9
    plain = IncompleteGCCA(n_components=k, shrinkage=0.5).fit(views)
    assert (
        principal_cosines(model.transform_shared(views), z).mean()
        > principal_cosines(plain.transform_shared(views), z).mean()
    )


def test_select_n_components_counts_joint_components() -> None:
    """Two shared factors are two components; no shared factor is none."""
    views, _ = _wide_factor_views(150, (300, 400, 200))
    assert IncompleteGCCA.select_n_components(views, random_state=0) == 2
    rng = np.random.default_rng(0)
    independent = [rng.standard_normal((150, p)) for p in (60, 80, 50)]
    assert IncompleteGCCA.select_n_components(independent, random_state=0) == 0


def test_select_n_components_is_reproducible() -> None:
    """The same seed gives the same answer."""
    views, _ = _wide_factor_views(100, (80, 90))
    first = IncompleteGCCA.select_n_components(views, random_state=4)
    assert first == IncompleteGCCA.select_n_components(views, random_state=4)
