"""QuantileCCA: canonical quantile regression."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, QuantileCCA
from tests._helpers import assert_same_scores_as, linear_views


@pytest.mark.parametrize("quantile", [0.25, 0.5])
def test_gaussian_data_give_cca_at_every_quantile(quantile: float) -> None:
    """With Gaussian errors each conditional quantile is a shifted mean: CCA."""
    train, test = linear_views(0, 1000), linear_views(1, 200)
    model = QuantileCCA(quantile=quantile, random_state=0).fit(train)
    reference = CCA().fit(train).transform(test)
    assert_same_scores_as(model.transform(test), reference, atol=2e-2)


def test_upper_quantile_follows_heteroscedastic_spread() -> None:
    """A response whose spread grows with x0 wins at 0.9 but not at the median."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, (1000, 2))
    Y = np.column_stack(
        [0.3 * X[:, 1] + rng.standard_normal(1000), X[:, 0] * rng.standard_normal(1000)]
    )
    median, upper = (
        np.abs(QuantileCCA(quantile=q, random_state=0).fit([X, Y]).weights_[1][:, 0])
        for q in (0.5, 0.9)
    )
    assert median[0] > median[1]
    assert upper[1] > upper[0]
