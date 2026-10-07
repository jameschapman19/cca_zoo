"""LightGBM -> XGBoost conversion, plus LightGBM estimators that only accept
parameters the conversion supports.

Semantic differences handled here:

* LightGBM sends ``x <= t`` left, XGBoost sends ``x < c`` left on float32
  data, so ``c`` is the float32 immediately above the largest float32 <= t.
* LightGBM folds the init score into the first tree and applies shrinkage to
  the leaves, so XGBoost's base score is set to the identity margin.
* LightGBM's categorical sets go left, XGBoost's go right, so children of
  categorical nodes are swapped.
"""

import functools

import lightgbm as lgb
import xgboost as xgb

from gbm2xgb import _xgb

# LightGBM objective -> (XGBoost objective, XGBoost base_score in output space)
_IDENTITY = {
    name: ("reg:squarederror", 0.0)
    for name in ("regression", "regression_l1", "huber", "fair", "quantile", "mape")
}
_LOG_LINK = {
    "poisson": ("count:poisson", 1.0),
    "gamma": ("reg:gamma", 1.0),
    "tweedie": ("reg:tweedie", 1.0),
}
REGRESSION_OBJECTIVES = (*_IDENTITY, *_LOG_LINK)
CLASSIFICATION_OBJECTIVES = ("binary", "multiclass")
BOOSTING_TYPES = ("gbdt", "dart", "goss")


def _objective(spec: str) -> tuple[str, float, dict]:
    """Return (xgboost objective, base_score, extras) for a LightGBM objective string."""
    name, *params = spec.split()
    opts = dict(p.split(":", 1) for p in params if ":" in p)
    if "sqrt" in params:
        raise NotImplementedError("regression with reg_sqrt is not supported")
    if name in _IDENTITY:
        return (*_IDENTITY[name], {})
    if name in _LOG_LINK:
        return (*_LOG_LINK[name], {})
    if name == "binary":
        return "binary:logistic", 0.5, {"sigmoid": float(opts["sigmoid"])}
    if name == "multiclass":
        return "multi:softprob", 0.0, {"num_class": int(opts["num_class"])}
    raise NotImplementedError(f"LightGBM objective {spec!r} is not supported")


def _convert_tree(tree: dict, leaf_scale: float, num_feature: int, tree_id: int) -> dict:
    t = _xgb.TreeBuilder()
    stack = [(tree["tree_structure"], t.root)]  # (lightgbm node, xgboost id)
    while stack:
        node, nid = stack.pop()

        if "leaf_value" in node:
            if "leaf_features" in node:
                raise NotImplementedError("linear trees are not supported")
            t.leaf(nid, node["leaf_value"] * leaf_scale, node["leaf_count"])
            continue

        lgb_left, lgb_right = node["left_child"], node["right_child"]
        missing = node["missing_type"]
        common = dict(gain=node["split_gain"], weight=node["internal_count"])

        if node["decision_type"] == "==":
            members = [int(c) for c in node["threshold"].split("||")]
            # LightGBM sends NaN (and negative or unseen categories) right; the
            # matching set goes left there but right in XGBoost, so swap.
            left, right = t.category_split(
                nid, node["split_feature"], members, default_left=True, **common
            )
            xgb_left, xgb_right = lgb_right, lgb_left
        elif node["decision_type"] == "<=":
            thr = node["threshold"]
            if missing == "Zero":
                raise NotImplementedError("zero_as_missing=True is not supported")
            # missing_type None maps NaN to 0.0 before comparing
            default_left = node["default_left"] if missing == "NaN" else 0.0 <= thr
            left, right = t.split(
                nid, node["split_feature"], _xgb.xgb_threshold(thr), default_left, **common
            )
            xgb_left, xgb_right = lgb_left, lgb_right
        else:
            raise NotImplementedError(f"decision_type {node['decision_type']!r}")

        stack.append((xgb_right, right))
        stack.append((xgb_left, left))
    return t.to_dict(tree_id, num_feature)


def to_xgboost_dict(model: lgb.Booster) -> dict:
    """Build the XGBoost JSON model document for a LightGBM booster
    (up to its best iteration, as ``model.predict`` uses)."""
    dump = model.dump_model()
    if dump["average_output"]:
        raise NotImplementedError("random forest (averaged) models are not supported")

    objective, base_score, extras = _objective(dump["objective"])
    num_feature = dump["max_feature_idx"] + 1
    leaf_scale = extras.get("sigmoid", 1.0)

    trees = [
        _convert_tree(t, leaf_scale, num_feature, i) for i, t in enumerate(dump["tree_info"])
    ]
    feature_types = ["float"] * num_feature
    for tree in trees:
        for nid in tree["categories_nodes"]:
            feature_types[tree["split_indices"][nid]] = "c"
    if "c" not in feature_types:
        feature_types = []
    # XGBoost rejects unnamed input when a model has names, so drop the
    # placeholder names LightGBM generates for numpy input.
    names = list(dump["feature_names"])
    if names == [f"Column_{i}" for i in range(num_feature)]:
        names = []

    return _xgb.booster_document(
        trees,
        trees_per_iteration=dump["num_tree_per_iteration"],
        objective=objective,
        base_score=base_score,
        num_class=extras.get("num_class", 0),
        num_feature=num_feature,
        feature_names=names,
        feature_types=feature_types,
    )


def _as_booster(model) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    if isinstance(model, lgb.LGBMModel):
        return model.booster_
    return lgb.Booster(model_file=str(model))


def convert(model) -> xgb.Booster:
    """Convert a LightGBM ``Booster``, sklearn-API model, or model file path
    to an ``xgboost.Booster``."""
    return _xgb.load(to_xgboost_dict(_as_booster(model)))


# --- estimators that reject what the conversion cannot represent -------------


def _check_params(params: dict, objectives: tuple[str, ...]) -> None:
    def require(name, allowed):
        value = params.get(name)
        if value is not None and value not in allowed:
            raise ValueError(
                f"{name}={value!r} cannot be converted to XGBoost; use one of {allowed}"
            )

    require("objective", objectives)
    for alias in ("boosting_type", "boosting"):
        require(alias, BOOSTING_TYPES)
    for flag in ("zero_as_missing", "linear_tree", "reg_sqrt"):
        require(flag, (False,))


class _Convertible:
    def _check(self):
        _check_params(self.get_params(), self._objectives)

    def set_params(self, **params):
        super().set_params(**params)
        self._check()
        return self

    def fit(self, X, y, sample_weight=None, init_score=None, **kwargs):
        # init_score is not stored in the model, so converted predictions would drop it
        if init_score is not None:
            raise ValueError("init_score cannot be converted to XGBoost")
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def to_xgboost(self) -> xgb.Booster:
        return convert(self)


class LGBMRegressor(_Convertible, lgb.LGBMRegressor):
    """``lightgbm.LGBMRegressor`` restricted to what ``to_xgboost`` supports."""

    _objectives = REGRESSION_OBJECTIVES

    @functools.wraps(lgb.LGBMRegressor.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()


class LGBMClassifier(_Convertible, lgb.LGBMClassifier):
    """``lightgbm.LGBMClassifier`` restricted to what ``to_xgboost`` supports."""

    _objectives = CLASSIFICATION_OBJECTIVES

    @functools.wraps(lgb.LGBMClassifier.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check()
