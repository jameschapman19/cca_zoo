import catboost as cb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import clone

import gbm2xgb
from gbm2xgb import catboost as gc


def make_data(n=1500, seed=0, nan_frac=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[rng.random(X.shape) < nan_frac] = np.nan
    y = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) ** 2 + rng.normal(scale=0.3, size=n)
    return X, y


def check(model, X, expected):
    got = gbm2xgb.convert(model).predict(xgb.DMatrix(X))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth", [1, 3, 6])
def test_regression(depth):
    X, y = make_data()
    m = cb.CatBoostRegressor(iterations=30, depth=depth, verbose=0).fit(X, y)
    check(m, X, m.predict(X))


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_nan_modes(nan_mode):
    X, y = make_data()
    m = cb.CatBoostRegressor(iterations=30, depth=4, nan_mode=nan_mode, verbose=0).fit(X, y)
    check(m, X, m.predict(X))


def test_no_nans_in_training_but_nans_at_predict():
    X, y = make_data(nan_frac=0.0)
    m = cb.CatBoostRegressor(iterations=30, depth=4, verbose=0).fit(X, y)
    Xn = X.copy()
    Xn[::5, 1] = np.nan
    check(m, Xn, m.predict(Xn))


def test_binary():
    X, y = make_data()
    m = cb.CatBoostClassifier(iterations=30, depth=4, verbose=0).fit(X, y > 0.5)
    check(m, X, m.predict_proba(X)[:, 1])


def test_multiclass():
    X, y = make_data()
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    m = cb.CatBoostClassifier(iterations=20, depth=4, loss_function="MultiClass", verbose=0).fit(X, labels)
    check(m, X, m.predict_proba(X))


def test_boost_from_average_bias():
    X, y = make_data()
    m = cb.CatBoostRegressor(iterations=20, depth=3, boost_from_average=True, verbose=0).fit(X, y + 100)
    check(m, X, m.predict(X))


def test_file_roundtrip(tmp_path):
    X, y = make_data()
    m = cb.CatBoostRegressor(iterations=10, depth=3, verbose=0).fit(X, y)
    m.save_model(tmp_path / "m.cbm")
    gc.convert(tmp_path / "m.cbm").save_model(tmp_path / "m.ubj")
    got = xgb.Booster(model_file=tmp_path / "m.ubj").predict(xgb.DMatrix(X))
    np.testing.assert_allclose(got, m.predict(X), rtol=1e-4, atol=1e-5)


def test_unsupported_models():
    X, y = make_data()
    lossguide = cb.CatBoostRegressor(iterations=5, grow_policy="Lossguide", verbose=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        gbm2xgb.convert(lossguide)
    Xc = pd.DataFrame(X[:, :2], columns=["a", "b"])
    Xc["c"] = np.random.default_rng(0).integers(0, 3, len(X))
    cat = cb.CatBoostRegressor(iterations=5, verbose=0).fit(Xc, y, cat_features=[2])
    with pytest.raises(NotImplementedError):
        gbm2xgb.convert(cat)


# --- restricted estimators ----------------------------------------------------


def test_regressor_wrapper():
    X, y = make_data()
    m = gc.CatBoostRegressor(iterations=20, verbose=0).fit(X, y)
    np.testing.assert_allclose(
        m.to_xgboost().predict(xgb.DMatrix(X)), m.predict(X), rtol=1e-4, atol=1e-5
    )


def test_classifier_wrapper():
    X, y = make_data()
    m = gc.CatBoostClassifier(iterations=20, verbose=0).fit(X, y > 0.5)
    np.testing.assert_allclose(
        m.to_xgboost().predict(xgb.DMatrix(X)), m.predict_proba(X)[:, 1], rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"grow_policy": "Lossguide"},
        {"grow_policy": "Depthwise"},
        {"loss_function": "Poisson"},
        {"loss_function": "MultiClass"},
        {"cat_features": [0]},
    ],
)
def test_regressor_blocks_unsupported_params(bad):
    with pytest.raises(ValueError):
        gc.CatBoostRegressor(**bad)
    with pytest.raises(ValueError):
        gc.CatBoostRegressor().set_params(**bad)


def test_classifier_blocks_unsupported_params():
    with pytest.raises(ValueError):
        gc.CatBoostClassifier(loss_function="MultiClassOneVsAll")
    with pytest.raises(ValueError):
        gc.CatBoostClassifier(loss_function="RMSE")


def test_fit_blocks_cat_features():
    X, y = make_data()
    with pytest.raises(ValueError):
        gc.CatBoostRegressor(iterations=2, verbose=0).fit(X, y, cat_features=[0])


def test_clone():
    m = gc.CatBoostRegressor(iterations=7, depth=3)
    c = clone(m)
    assert isinstance(c, gc.CatBoostRegressor)
    assert c.get_params() == m.get_params()
