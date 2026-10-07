import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lgb2xgb


def make_data(n=2000, seed=0, categorical=False, nan_frac=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    if categorical:
        X[:, 4] = rng.integers(0, 6, n)
    X[rng.random(X.shape) < nan_frac] = np.nan
    if categorical:
        X[:, 4] = np.where(np.isnan(X[:, 4]), 2, X[:, 4])
    signal = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) ** 2
    if categorical:
        signal = signal + (X[:, 4] % 3 == 1)
    return X, signal + rng.normal(scale=0.3, size=n)


def fit(params, X, y, rounds=30, categorical=None):
    ds = lgb.Dataset(X, y, categorical_feature=categorical or "auto")
    return lgb.train({"verbose": -1, "num_leaves": 15, **params}, ds, rounds)


def assert_same(booster, X, **kw):
    expected = booster.predict(X)
    got = lgb2xgb.convert(booster).predict(xgb.DMatrix(X))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5, **kw)


@pytest.mark.parametrize("objective", ["regression", "regression_l1", "huber"])
def test_regression(objective):
    X, y = make_data()
    assert_same(fit({"objective": objective}, X, y), X)


def test_binary():
    X, y = make_data()
    assert_same(fit({"objective": "binary"}, X, (y > np.median(y)) * 1.0), X)


def test_binary_sigmoid_scale():
    X, y = make_data()
    assert_same(fit({"objective": "binary", "sigmoid": 2.5}, X, (y > 0) * 1.0), X)


def test_multiclass():
    X, y = make_data()
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    b = fit({"objective": "multiclass", "num_class": 3}, X, labels)
    assert_same(b, X)


@pytest.mark.parametrize("objective", ["poisson", "gamma", "tweedie"])
def test_log_link(objective):
    X, y = make_data()
    assert_same(fit({"objective": objective}, X, np.exp(y / 3)), X)


def test_missing_type_none():
    X, y = make_data(nan_frac=0.0)
    b = fit({"objective": "regression"}, X, y)
    Xn = X.copy()
    Xn[::7, 0] = np.nan
    assert_same(b, Xn)


def test_categorical():
    X, y = make_data(categorical=True)
    b = fit({"objective": "regression"}, X, y, categorical=[4])
    Xn = X.copy()
    Xn[::5, 4] = np.nan
    Xn[::11, 4] = 9  # unseen category
    expected = b.predict(Xn)
    bst = lgb2xgb.convert(b)
    got = bst.predict(xgb.DMatrix(Xn, feature_types=["float"] * 4 + ["c"]))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_threshold_boundaries():
    # data on a coarse grid so values land exactly on and next to thresholds
    rng = np.random.default_rng(1)
    X = rng.integers(-5, 5, size=(2000, 3)).astype(np.float32)
    y = X[:, 0] * (X[:, 1] > 0) + rng.normal(scale=0.1, size=2000)
    b = fit({"objective": "regression"}, X, y)
    grid = np.concatenate([X, X + np.float32(1e-6), X - np.float32(1e-6)]).astype(np.float32)
    assert_same(b, grid)


def test_sklearn_wrapper_and_file_roundtrip(tmp_path):
    X, y = make_data()
    reg = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    expected = reg.predict(X)

    np.testing.assert_allclose(
        lgb2xgb.convert(reg).predict(xgb.DMatrix(X)), expected, rtol=1e-4, atol=1e-5
    )

    reg.booster_.save_model(tmp_path / "m.txt")
    lgb2xgb.convert_file(tmp_path / "m.txt", tmp_path / "m.ubj")
    loaded = xgb.Booster(model_file=tmp_path / "m.ubj")
    np.testing.assert_allclose(loaded.predict(xgb.DMatrix(X)), expected, rtol=1e-4, atol=1e-5)


def test_single_leaf_trees():
    X, y = make_data()
    b = fit({"objective": "regression", "min_data_in_leaf": 5000}, X, y, rounds=3)
    assert_same(b, X)


def test_unsupported():
    X, y = make_data()
    with pytest.raises(NotImplementedError):
        lgb2xgb.convert(fit({"objective": "regression", "boosting": "rf", "bagging_fraction": 0.5, "bagging_freq": 1}, X, y))
    with pytest.raises(NotImplementedError):
        lgb2xgb.convert(fit({"objective": "cross_entropy"}, X, (y > 0) * 1.0))
    with pytest.raises(NotImplementedError):
        lgb2xgb.convert(fit({"objective": "regression", "zero_as_missing": True}, X, y))
