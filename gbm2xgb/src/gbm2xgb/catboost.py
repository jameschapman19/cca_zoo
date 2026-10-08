"""CatBoost -> XGBoost conversion, plus CatBoost estimators that only accept
parameters the conversion supports.

Only numerical features and symmetric (oblivious) trees are supported. Each
oblivious tree of depth d becomes a full binary XGBoost tree with 2**d leaves.
CatBoost sends ``x > border`` right and folds ``scale * sum + bias`` into the
prediction; the scale is applied to the leaves and the bias to the first
iteration's leaves.
"""

import functools
import json
import tempfile
from pathlib import Path

import catboost as cb
import numpy as np
import xgboost as xgb

from gbm2xgb import _xgb

_IDENTITY_LOSSES = ("RMSE", "MAE", "Quantile", "Huber", "Expectile", "LogCosh", "MAPE")
REGRESSION_LOSSES = _IDENTITY_LOSSES
CLASSIFICATION_LOSSES = ("Logloss", "CrossEntropy", "MultiClass")
UNSUPPORTED_FIT_ARGS = ("cat_features", "text_features", "embedding_features", "baseline")


def _objective(loss: str) -> str:
    if loss in _IDENTITY_LOSSES:
        return "reg:squarederror"
    if loss in ("Logloss", "CrossEntropy"):
        return "binary:logistic"
    if loss == "MultiClass":
        return "multi:softprob"
    raise NotImplementedError(f"CatBoost loss {loss!r} is not supported")


def _convert_oblivious_tree(
    splits: list[dict],
    leaf_values: np.ndarray,
    leaf_weights: list[float],
    feature_index: list[int],
    default_left: list[bool],
    num_feature: int,
    tree_id: int,
) -> dict:
    """``leaf_values`` is the (2**d,) leaf vector for one output dimension."""
    depth = len(splits)
    t = _xgb.TreeBuilder()
    # CatBoost's leaf index has bit i set iff x[splits[i]] > border_i
    stack = [(t.root, 0, 0)]  # (xgboost id, level from root, leaf-index bits so far)
    while stack:
        nid, level, bits = stack.pop()
        if level == depth:
            t.leaf(nid, leaf_values[bits], leaf_weights[bits])
            continue
        bit = depth - 1 - level
        s = splits[bit]
        feature = feature_index[s["float_feature_index"]]
        left, right = t.split(
            nid,
            feature,
            _xgb.xgb_threshold(s["border"]),
            default_left[s["float_feature_index"]],
        )
        stack.append((right, level + 1, bits | 1 << bit))
        stack.append((left, level + 1, bits))
    return t.to_dict(tree_id, num_feature)


def to_xgboost_dict(model: cb.CatBoost) -> dict:
    """Build the XGBoost JSON model document for a fitted CatBoost model."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.json"
        model.save_model(path, format="json")
        doc = json.loads(path.read_text())

    if "oblivious_trees" not in doc:
        raise NotImplementedError("only symmetric trees (grow_policy='SymmetricTree') are supported")
    info = doc["features_info"]
    if set(info) != {"float_features"}:
        raise NotImplementedError("categorical, text and embedding features are not supported")
    floats = info["float_features"]
    feature_index = [f["flat_feature_index"] for f in floats]
    num_feature = len(floats)

    treatments = {f["nan_value_treatment"] for f in floats}
    if not treatments <= {"AsFalse", "AsTrue", "AsIs"}:
        raise NotImplementedError(f"nan_value_treatment {treatments - {'AsFalse', 'AsTrue', 'AsIs'}}")
    # AsFalse/AsIs: NaN compares as not-greater, i.e. goes left
    default_left = [f["nan_value_treatment"] != "AsTrue" for f in floats]

    scale, bias = doc["scale_and_bias"]
    bias = np.atleast_1d(np.asarray(bias, dtype=float))
    loss = doc["model_info"]["params"]["loss_function"]["type"]

    trees = []
    dim = None
    for i, tree in enumerate(doc["oblivious_trees"]):
        leaves = np.asarray(tree["leaf_values"], dtype=float) * scale
        dim = len(leaves) >> len(tree["splits"])
        leaves = leaves.reshape(-1, dim)  # leaf-major: leaf * dim + output
        for c in range(dim):
            values = leaves[:, c] + (bias[c] if i == 0 else 0.0)
            trees.append(
                _convert_oblivious_tree(
                    tree["splits"],
                    values,
                    tree["leaf_weights"],
                    feature_index,
                    default_left,
                    num_feature,
                    len(trees),
                )
            )

    objective = _objective(loss)
    names = [f["feature_id"] for f in floats]
    if not all(names):
        names = []
    return _xgb.booster_document(
        trees,
        trees_per_iteration=dim,
        objective=objective,
        num_class=dim if objective == "multi:softprob" else 0,
        num_feature=num_feature,
        feature_names=names,
        feature_types=[],
    )


def convert(model) -> xgb.Booster:
    """Convert a fitted CatBoost model, or a CatBoost model file path, to an
    ``xgboost.Booster``."""
    if not isinstance(model, cb.CatBoost):
        model = cb.CatBoost().load_model(str(model))
    return _xgb.load(to_xgboost_dict(model))


# --- estimators that reject what the conversion cannot represent -------------


def _check_params(params: dict, losses: tuple[str, ...]) -> None:
    loss = params.get("loss_function")
    if loss is not None and (not isinstance(loss, str) or loss.split(":")[0] not in losses):
        raise ValueError(
            f"loss_function={loss!r} cannot be converted to XGBoost; use one of {losses}"
        )
    policy = params.get("grow_policy")
    if policy not in (None, "SymmetricTree"):
        raise ValueError(f"grow_policy={policy!r} cannot be converted; use 'SymmetricTree'")
    for name in UNSUPPORTED_FIT_ARGS:
        if params.get(name) is not None:
            raise ValueError(f"{name} cannot be converted to XGBoost; only numerical features are supported")


class _Convertible:
    def _check(self):
        _check_params(self.get_params(), self._losses)

    def set_params(self, **params):
        super().set_params(**params)
        self._check()
        return self

    def fit(self, X, y=None, *args, **kwargs):
        # positional args after y are cat_features, text_features, embedding_features
        if any(a is not None for a in args[:3]) or any(
            kwargs.get(name) is not None for name in UNSUPPORTED_FIT_ARGS
        ):
            raise ValueError(
                "cat_features, text_features, embedding_features and baseline "
                "cannot be converted to XGBoost"
            )
        return super().fit(X, y, *args, **kwargs)

    def to_xgboost(self):
        return _xgb.to_sklearn(convert(self), self._xgb_estimator, getattr(self, "classes_", None))


class CatBoostRegressor(_Convertible, cb.CatBoostRegressor):
    """``catboost.CatBoostRegressor`` restricted to what ``to_xgboost`` supports."""

    _losses = REGRESSION_LOSSES
    _xgb_estimator = xgb.XGBRegressor

    @functools.wraps(cb.CatBoostRegressor.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()


class CatBoostClassifier(_Convertible, cb.CatBoostClassifier):
    """``catboost.CatBoostClassifier`` restricted to what ``to_xgboost`` supports."""

    _losses = CLASSIFICATION_LOSSES
    _xgb_estimator = xgb.XGBClassifier

    @functools.wraps(cb.CatBoostClassifier.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()
