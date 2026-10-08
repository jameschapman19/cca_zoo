"""fit a wrapped estimator -> save .ubj -> load it as a plain XGBoost estimator."""

import catboost as cb  # noqa: F401  (ensure installed)
import numpy as np
import pytest
import xgboost as xgb

from gbm2xgb import catboost as gc
from gbm2xgb import lightgbm as gl


def data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[rng.random(X.shape) < 0.08] = np.nan
    y = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) ** 2 + rng.normal(scale=0.3, size=n)
    return X, y


@pytest.mark.parametrize(
    "make", [lambda: gl.LGBMRegressor(n_estimators=30, verbose=-1),
             lambda: gc.CatBoostRegressor(iterations=30, verbose=0, allow_writing_files=False)]
)
def test_regressor(make, tmp_path):
    X, y = data()
    model = make().fit(X, y)
    model.to_xgboost().save_model(tmp_path / "m.ubj")

    loaded = xgb.XGBRegressor()
    loaded.load_model(tmp_path / "m.ubj")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize(
    "make", [lambda: gl.LGBMClassifier(n_estimators=30, verbose=-1),
             lambda: gc.CatBoostClassifier(iterations=30, verbose=0, allow_writing_files=False)]
)
def test_classifier(make, n_classes, tmp_path):
    X, y = data()
    labels = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_classes + 1)[1:-1]))
    model = make().fit(X, labels)
    model.to_xgboost().save_model(tmp_path / "m.ubj")

    loaded = xgb.XGBClassifier()
    loaded.load_model(tmp_path / "m.ubj")
    assert loaded.n_classes_ == n_classes
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(loaded.predict(X), np.ravel(model.predict(X)))
