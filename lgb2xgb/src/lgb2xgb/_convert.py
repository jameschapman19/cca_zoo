"""LightGBM -> XGBoost model conversion.

Semantic differences handled here:

* LightGBM sends ``x <= t`` left, XGBoost sends ``x < c`` left on float32
  data, so ``c`` is the float32 immediately above the largest float32 <= t.
* LightGBM folds the init score into the first tree and applies shrinkage to
  the leaves, so XGBoost's base score is set to the identity margin.
* LightGBM's categorical sets go left, XGBoost's go right, so children of
  categorical nodes are swapped.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

_F32_MAX = float(np.finfo(np.float32).max)
_NO_PARENT = 2147483647

# LightGBM objective -> (XGBoost objective, XGBoost base_score in output space)
_IDENTITY = {
    name: ("reg:squarederror", 0.0)
    for name in (
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "quantile",
        "mape",
    )
}
_LOG_LINK = {
    "poisson": ("count:poisson", 1.0),
    "gamma": ("reg:gamma", 1.0),
    "tweedie": ("reg:tweedie", 1.0),
}


def _xgb_threshold(t: float) -> float:
    """Smallest float32 c such that for every float32 x: x <= t  <=>  x < c."""
    with np.errstate(over="ignore"):
        f = np.float32(t)
        if f > t:
            f = np.nextafter(f, np.float32(-np.inf))
        c = float(np.nextafter(f, np.float32(np.inf)))
    return min(max(c, -_F32_MAX), _F32_MAX)


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
    left, right, parent = [], [], []
    split_index, split_cond, default_left, split_type = [], [], [], []
    gain, hess = [], []
    cat_nodes, cat_segments, cat_sizes, cats = [], [], [], []

    # XGBoost allocates the two children of a node as consecutive ids, and
    # parts of its predictor rely on that, so allocate them as a pair.
    def new_node(parent_id):
        left.append(-1)
        right.append(-1)
        parent.append(parent_id)
        split_index.append(0)
        split_cond.append(0.0)
        default_left.append(0)
        split_type.append(0)
        gain.append(0.0)
        hess.append(0.0)
        return len(left) - 1

    stack = [(tree["tree_structure"], new_node(_NO_PARENT))]  # (lightgbm node, xgboost id)
    while stack:
        node, nid = stack.pop()

        if "leaf_value" in node:
            if "leaf_features" in node:
                raise NotImplementedError("linear trees are not supported")
            split_cond[nid] = float(np.float32(node["leaf_value"] * leaf_scale))
            hess[nid] = float(node["leaf_count"])
            continue

        split_index[nid] = node["split_feature"]
        gain[nid] = float(node["split_gain"])
        hess[nid] = float(node["internal_count"])
        missing = node["missing_type"]
        lgb_left, lgb_right = node["left_child"], node["right_child"]

        if node["decision_type"] == "==":
            members = sorted(int(c) for c in node["threshold"].split("||"))
            cat_nodes.append(nid)
            cat_segments.append(len(cats))
            cat_sizes.append(len(members))
            cats.extend(members)
            split_type[nid] = 1
            # LightGBM sends NaN (and negative or unseen categories) right
            default_left[nid] = 1
            lgb_left, lgb_right = lgb_right, lgb_left
        elif node["decision_type"] == "<=":
            t = node["threshold"]
            if missing == "Zero":
                raise NotImplementedError("zero_as_missing=True is not supported")
            split_cond[nid] = _xgb_threshold(t)
            # missing_type None maps NaN to 0.0 before comparing
            default_left[nid] = int(node["default_left"] if missing == "NaN" else 0.0 <= t)
        else:
            raise NotImplementedError(f"decision_type {node['decision_type']!r}")

        left[nid] = new_node(nid)
        right[nid] = new_node(nid)
        stack.append((lgb_right, right[nid]))
        stack.append((lgb_left, left[nid]))

    n = len(left)
    return {
        "base_weights": [c if l == -1 else 0.0 for c, l in zip(split_cond, left)],
        "categories": cats,
        "categories_nodes": cat_nodes,
        "categories_segments": cat_segments,
        "categories_sizes": cat_sizes,
        "default_left": default_left,
        "id": tree_id,
        "left_children": left,
        "loss_changes": gain,
        "parents": parent,
        "right_children": right,
        "split_conditions": split_cond,
        "split_indices": split_index,
        "split_type": split_type,
        "sum_hessian": hess,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": str(num_feature),
            "num_nodes": str(n),
            "size_leaf_vector": "1",
        },
    }


def to_xgboost_dict(model: lgb.Booster) -> dict:
    """Build the XGBoost JSON model document for a LightGBM booster."""
    dump = model.dump_model(num_iteration=-1)
    if dump["average_output"]:
        raise NotImplementedError("random forest (averaged) models are not supported")

    objective, base_score, extras = _objective(dump["objective"])
    k = dump["num_tree_per_iteration"]
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

    obj = {"name": objective}
    if objective == "binary:logistic":
        obj["reg_loss_param"] = {"scale_pos_weight": "1"}
    elif objective == "multi:softprob":
        obj["softmax_multiclass_param"] = {"num_class": str(extras["num_class"])}
    elif objective == "count:poisson":
        obj["poisson_regression_param"] = {"max_delta_step": "0.7"}
    elif objective == "reg:tweedie":
        obj["tweedie_regression_param"] = {"tweedie_variance_power": "1.5"}

    return {
        "version": [3, 0, 0],
        "learner": {
            "attributes": {},
            "feature_names": names,
            "feature_types": feature_types,
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(len(trees)),
                    },
                    "iteration_indptr": list(range(0, len(trees) + 1, k)),
                    "tree_info": [i % k for i in range(len(trees))],
                    "trees": trees,
                },
            },
            "learner_model_param": {
                "base_score": f"[{base_score!r}]",
                "boost_from_average": "1",
                "num_class": str(extras.get("num_class", 0)),
                "num_feature": str(num_feature),
                "num_target": "1",
            },
            "objective": obj,
        },
    }


def _as_booster(model) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    if isinstance(model, lgb.LGBMModel):
        return model.booster_
    return lgb.Booster(model_file=str(model))


def convert(model) -> xgb.Booster:
    """Convert a LightGBM ``Booster``, sklearn-API model, or model file path
    to an ``xgboost.Booster``."""
    doc = to_xgboost_dict(_as_booster(model))
    return xgb.Booster(model_file=bytearray(json.dumps(doc).encode()))


def convert_file(src, dst) -> None:
    """Convert a LightGBM model file; ``dst`` ending in ``.ubj`` is XGBoost's
    binary format, ``.json`` its JSON format."""
    dst = Path(dst)
    if dst.suffix not in (".json", ".ubj"):
        raise ValueError(f"dst must end in .json or .ubj, got {dst.suffix!r}")
    convert(src).save_model(dst)
