"""Feature names and output containers beyond a single model."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.base import clone

from cca_zoo.linear import CCA, RidgeCCA
from cca_zoo.model_selection import GridSearchCV, cross_val_predict
from tests._helpers import linear_views


def _frames(views: list[np.ndarray]) -> list[pd.DataFrame]:
    return [
        pd.DataFrame(v, columns=[f"v{i}_{j}" for j in range(v.shape[1])])
        for i, v in enumerate(views)
    ]


def test_feature_names_out_follow_sklearn() -> None:
    """Latent dimensions are named <model><k> in every view, as sklearn's PCA."""
    model = CCA(2).fit(linear_views(0, 50))
    assert [list(n) for n in model.get_feature_names_out()] == [["cca0", "cca1"]] * 2


def test_polars_output() -> None:
    """set_output(transform="polars") gives polars DataFrames."""
    views = linear_views(0, 50)
    scores = CCA(2).set_output(transform="polars").fit(views).transform(views)
    assert all(isinstance(s, pl.DataFrame) for s in scores)
    assert scores[0].columns == ["cca0", "cca1"]


def test_global_config_and_clone_keep_the_container() -> None:
    """Sklearn's transform_output config applies, and clone keeps set_output."""
    views = linear_views(0, 50)
    with sklearn.config_context(transform_output="pandas"):
        assert isinstance(CCA(2).fit(views).transform(views)[0], pd.DataFrame)
    cloned = clone(CCA(2).set_output(transform="pandas"))
    assert isinstance(cloned.fit(views).transform(views)[0], pd.DataFrame)


def test_refit_on_arrays_forgets_names() -> None:
    """A refit without names drops the names of the earlier fit."""
    views = linear_views(0, 50)
    model = CCA().fit(_frames(views)).fit(views)
    assert not hasattr(model, "feature_names_per_view_")


def test_search_keeps_feature_names() -> None:
    """Searches and cross-validation pass each view's names to the model."""
    frames = _frames(linear_views(0, 60))
    search = GridSearchCV(RidgeCCA(), {"shrinkage": [0.1, 0.5]}, cv=3).fit(frames)
    assert [list(n) for n in search.best_estimator_.feature_names_per_view_] == [
        list(f.columns) for f in frames
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        search.transform(frames)
        search.score(frames)
        cross_val_predict(RidgeCCA(), frames, cv=3)
