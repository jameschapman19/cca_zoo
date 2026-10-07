import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import gbm2xgb
from gbm2xgb import lightgbm as gl


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
    got = gbm2xgb.convert(booster).predict(xgb.DMatrix(X))
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
    bst = gbm2xgb.convert(b)
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
        gbm2xgb.convert(reg).predict(xgb.DMatrix(X)), expected, rtol=1e-4, atol=1e-5
    )

    reg.booster_.save_model(tmp_path / "m.txt")
    gl.convert(tmp_path / "m.txt").save_model(tmp_path / "m.ubj")
    loaded = xgb.Booster(model_file=tmp_path / "m.ubj")
    np.testing.assert_allclose(loaded.predict(xgb.DMatrix(X)), expected, rtol=1e-4, atol=1e-5)


def test_single_leaf_trees():
    X, y = make_data()
    b = fit({"objective": "regression", "min_data_in_leaf": 5000}, X, y, rounds=3)
    assert_same(b, X)


def test_unsupported():
    X, y = make_data()
    with pytest.raises(NotImplementedError):
        gbm2xgb.convert(fit({"objective": "regression", "boosting": "rf", "bagging_fraction": 0.5, "bagging_freq": 1}, X, y))
    with pytest.raises(NotImplementedError):
        gbm2xgb.convert(fit({"objective": "cross_entropy"}, X, (y > 0) * 1.0))
    with pytest.raises(NotImplementedError):
        gbm2xgb.convert(fit({"objective": "regression", "zero_as_missing": True}, X, y))


def test_early_stopping_uses_best_iteration():
    X, y = make_data()
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.3, verbose=-1)
    reg.fit(X[:1000], y[:1000], eval_set=[(X[1000:], y[1000:])],
            callbacks=[lgb.early_stopping(5, verbose=False)])
    assert reg.best_iteration_ < 300
    np.testing.assert_allclose(
        gbm2xgb.convert(reg).predict(xgb.DMatrix(X)), reg.predict(X), rtol=1e-4, atol=1e-5
    )


# --- restricted estimators ----------------------------------------------------


def test_regressor_wrapper():
    X, y = make_data()
    reg = gl.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    xreg = reg.to_xgboost()
    assert isinstance(xreg, xgb.XGBRegressor)
    np.testing.assert_allclose(xreg.predict(X), reg.predict(X), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n_classes", [2, 4])
def test_classifier_wrapper(n_classes):
    X, y = make_data()
    labels = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_classes + 1)[1:-1]))
    clf = gl.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, labels)
    xclf = clf.to_xgboost()
    assert isinstance(xclf, xgb.XGBClassifier)
    np.testing.assert_allclose(xclf.predict_proba(X), clf.predict_proba(X), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(xclf.predict(X), clf.predict(X))


def test_classifier_with_non_index_labels_is_blocked():
    X, y = make_data()
    clf = gl.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, np.where(y > 0, 5, 7))
    with pytest.raises(ValueError):
        clf.to_xgboost()


@pytest.mark.parametrize(
    "bad",
    [
        {"objective": "cross_entropy"},
        {"objective": "multiclass"},
        {"objective": lambda y, p: (p - y, p * 0 + 1)},
        {"boosting_type": "rf"},
        {"zero_as_missing": True},
        {"linear_tree": True},
        {"reg_sqrt": True},
    ],
)
def test_regressor_blocks_unsupported_params(bad):
    with pytest.raises(ValueError):
        gl.LGBMRegressor(**bad)
    with pytest.raises(ValueError):
        gl.LGBMRegressor().set_params(**bad)


def test_classifier_blocks_unsupported_params():
    with pytest.raises(ValueError):
        gl.LGBMClassifier(objective="multiclassova")
    with pytest.raises(ValueError):
        gl.LGBMClassifier(objective="regression")


def test_fit_blocks_init_score():
    X, y = make_data()
    with pytest.raises(ValueError):
        gl.LGBMRegressor(verbose=-1).fit(X, y, init_score=np.zeros(len(y)))


def test_sklearn_clone_and_params_roundtrip():
    from sklearn.base import clone

    reg = gl.LGBMRegressor(n_estimators=7, objective="huber", reg_lambda=0.5)
    cloned = clone(reg)
    assert isinstance(cloned, gl.LGBMRegressor)
    assert cloned.get_params() == reg.get_params()
