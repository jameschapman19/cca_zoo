"""Shared pieces for building an XGBoost JSON model document."""

import json
import math

import numpy as np
import xgboost as xgb

_F32_MAX = float(np.finfo(np.float32).max)
_NO_PARENT = 2147483647


def xgb_threshold(t: float) -> float:
    """Smallest float32 c such that for every float32 x: x <= t  <=>  x < c."""
    with np.errstate(over="ignore"):
        f = np.float32(t)
        if float(f) > t:  # compare in double; numpy would cast t to float32
            f = np.nextafter(f, np.float32(-np.inf))
        c = float(np.nextafter(f, np.float32(np.inf)))
    return min(max(c, -_F32_MAX), _F32_MAX)


class TreeBuilder:
    """Builds one XGBoost tree. XGBoost allocates the two children of a node as
    consecutive ids and parts of its predictor rely on that, so children are
    only ever added as a pair."""

    def __init__(self):
        self.left, self.right, self.parent = [], [], []
        self.split_index, self.split_cond, self.default_left = [], [], []
        self.split_type, self.gain, self.hess = [], [], []
        self.cat_nodes, self.cat_segments, self.cat_sizes, self.cats = [], [], [], []
        self._new_node(_NO_PARENT)

    @property
    def root(self) -> int:
        return 0

    def _new_node(self, parent_id: int) -> int:
        self.left.append(-1)
        self.right.append(-1)
        self.parent.append(parent_id)
        self.split_index.append(0)
        self.split_cond.append(0.0)
        self.default_left.append(0)
        self.split_type.append(0)
        self.gain.append(0.0)
        self.hess.append(0.0)
        return len(self.left) - 1

    def leaf(self, nid: int, value: float, weight: float = 0.0) -> None:
        self.split_cond[nid] = float(np.float32(value))
        self.hess[nid] = float(weight)

    def split(
        self, nid: int, feature: int, cond: float, default_left: bool, gain=0.0, weight=0.0
    ) -> tuple[int, int]:
        """Make ``nid`` a numerical split (left iff x < cond); returns (left, right) ids."""
        self.split_index[nid] = feature
        self.split_cond[nid] = cond
        self.default_left[nid] = int(default_left)
        self.gain[nid] = float(gain)
        self.hess[nid] = float(weight)
        self.left[nid] = self._new_node(nid)
        self.right[nid] = self._new_node(nid)
        return self.left[nid], self.right[nid]

    def category_split(
        self, nid: int, feature: int, right_set: list[int], default_left: bool, gain=0.0, weight=0.0
    ) -> tuple[int, int]:
        """Make ``nid`` a categorical split (right iff category in ``right_set``)."""
        self.cat_nodes.append(nid)
        self.cat_segments.append(len(self.cats))
        self.cat_sizes.append(len(right_set))
        self.cats.extend(sorted(right_set))
        self.split_type[nid] = 1
        return self.split(nid, feature, 0.0, default_left, gain, weight)

    def add_to_leaves(self, bias: float) -> None:
        for nid, l in enumerate(self.left):
            if l == -1:
                self.split_cond[nid] = float(np.float32(self.split_cond[nid] + bias))

    def _sorted_categories(self):
        """XGBoost requires categorical records ordered by ascending node id."""
        segments = [
            (nid, self.cats[start : start + size])
            for nid, start, size in zip(self.cat_nodes, self.cat_segments, self.cat_sizes)
        ]
        segments.sort(key=lambda s: s[0])
        nodes = [nid for nid, _ in segments]
        sizes = [len(c) for _, c in segments]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
        flat = [c for _, cs in segments for c in cs]
        return nodes, starts, sizes, flat

    def to_dict(self, tree_id: int, num_feature: int) -> dict:
        cat_nodes, cat_segments, cat_sizes, cats = self._sorted_categories()
        return {
            "base_weights": [c if l == -1 else 0.0 for c, l in zip(self.split_cond, self.left)],
            "categories": cats,
            "categories_nodes": cat_nodes,
            "categories_segments": cat_segments,
            "categories_sizes": cat_sizes,
            "default_left": self.default_left,
            "id": tree_id,
            "left_children": self.left,
            "loss_changes": self.gain,
            "parents": self.parent,
            "right_children": self.right,
            "split_conditions": self.split_cond,
            "split_indices": self.split_index,
            "split_type": self.split_type,
            "sum_hessian": self.hess,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(num_feature),
                "num_nodes": str(len(self.left)),
                "size_leaf_vector": "1",
            },
        }


def _objective_block(name: str, num_class: int) -> dict:
    obj = {"name": name}
    if name == "binary:logistic":
        obj["reg_loss_param"] = {"scale_pos_weight": "1"}
    elif name == "multi:softprob":
        obj["softmax_multiclass_param"] = {"num_class": str(num_class)}
    elif name == "count:poisson":
        obj["poisson_regression_param"] = {"max_delta_step": "0.7"}
    elif name == "reg:tweedie":
        obj["tweedie_regression_param"] = {"tweedie_variance_power": "1.5"}
    return obj


# XGBoost releases before 3.1 cannot parse the array form of base_score that 3.1+
# writes and silently fall back to 0.5, so the document always uses 0.5 and the
# difference to a zero margin is folded into the first tree.
_BASE_SCORE = 0.5
_BASE_MARGIN = {  # margin that base_score=0.5 contributes, per objective
    "reg:squarederror": 0.5,
    "binary:logistic": 0.0,
    "multi:softprob": 0.0,  # constant shift, cancelled by the softmax
    "count:poisson": math.log(0.5),
    "reg:gamma": math.log(0.5),
    "reg:tweedie": math.log(0.5),
}


def _shift_leaves(tree: dict, delta: float) -> None:
    for i, left in enumerate(tree["left_children"]):
        if left == -1:
            value = float(np.float32(tree["split_conditions"][i] + delta))
            tree["split_conditions"][i] = tree["base_weights"][i] = value


def booster_document(
    trees: list[dict],
    *,
    trees_per_iteration: int,
    objective: str,
    num_class: int,
    num_feature: int,
    feature_names: list[str],
    feature_types: list[str],
) -> dict:
    """The XGBoost JSON document, whose predictions are the plain sum of the leaves
    (through the objective's link)."""
    k = trees_per_iteration
    if trees:
        _shift_leaves(trees[0], -_BASE_MARGIN[objective])
    return {
        "version": [3, 0, 0],
        "learner": {
            "attributes": {},
            "feature_names": feature_names,
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
                "base_score": f"{_BASE_SCORE!r}",
                "boost_from_average": "1",
                "num_class": str(num_class),
                "num_feature": str(num_feature),
                "num_target": "1",
            },
            "objective": _objective_block(objective, num_class),
        },
    }


def load(doc: dict) -> xgb.Booster:
    return xgb.Booster(model_file=bytearray(json.dumps(doc).encode()))


def to_sklearn(booster: xgb.Booster, estimator: type, classes=None):
    """Wrap ``booster`` in ``XGBRegressor`` / ``XGBClassifier``. XGBClassifier
    predicts class indices, so the source classes must already be 0..k-1."""
    if classes is not None and not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            f"classes_={list(classes)} must be 0..{len(classes) - 1} to convert to XGBClassifier; "
            "encode the labels first"
        )
    model = estimator()
    model.load_model(booster.save_raw("ubj"))
    return model
