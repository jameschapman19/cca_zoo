"""The Array API models compute in the namespace of their inputs.

Needs ``SCIPY_ARRAY_API=1`` set before scipy is imported, so CI runs this
module in a step of its own.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import sklearn
from sklearn.utils._array_api import _convert_to_numpy, get_namespace

from cca_zoo._base import BaseModel
from cca_zoo.linear import CCA, GCCA, MCCA, PLS, TCCA, RidgeCCA
from tests._helpers import linear_views

pytestmark = pytest.mark.skipif(
    os.environ.get("SCIPY_ARRAY_API") != "1",
    reason="sklearn's array_api_dispatch needs SCIPY_ARRAY_API=1",
)

_SUPPORTED = [
    CCA(2),
    RidgeCCA(2, shrinkage=0.3),
    PLS(2),
    MCCA(2, shrinkage=0.2),
    MCCA(2, shrinkage=0.2, pca=False),
    GCCA(2, shrinkage=0.2),
]


@pytest.mark.parametrize("namespace", ["array_api_strict", "torch"])
@pytest.mark.parametrize("model", _SUPPORTED, ids=repr)
def test_array_api_inputs_give_numpy_results(model: BaseModel, namespace: str) -> None:
    """Fitted on another namespace, a model returns its numpy scores, in kind."""
    xp = pytest.importorskip(namespace)
    train, test = linear_views(0, 50), linear_views(1, 20)
    expected = model.fit(train).transform(test)
    with sklearn.config_context(array_api_dispatch=True):
        scores = model.fit([xp.asarray(v) for v in train]).transform(
            [xp.asarray(v) for v in test]
        )
        assert all(get_namespace(s)[0].__name__.endswith(namespace) for s in scores)
        scores = [_convert_to_numpy(s, get_namespace(s)[0]) for s in scores]
    for x, y in zip(scores, expected):
        signs = np.sign(np.sum(x * y, axis=0))
        np.testing.assert_allclose(x, y * signs, atol=1e-8)


def test_numpy_only_models_refuse_other_namespaces() -> None:
    """A model without Array API support says so rather than failing in numpy."""
    xp = pytest.importorskip("array_api_strict")
    views = [xp.asarray(v) for v in linear_views(0, 50)]
    with (
        sklearn.config_context(array_api_dispatch=True),
        pytest.raises(TypeError, match="computes with numpy"),
    ):
        TCCA().fit(views)
