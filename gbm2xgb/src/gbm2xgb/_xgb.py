"""Shared pieces for building an XGBoost JSON model document."""

import json

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

    def to_dict(self, tree_id: int, num_feature: int) -> dict:
        return {
            "base_weights": [c if l == -1 else 0.0 for c, l in zip(self.split_cond, self.left)],
            "categories": self.cats,
            "categories_nodes": self.cat_nodes,
            "categories_segments": self.cat_segments,
            "categories_sizes": self.cat_sizes,
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


def booster_document(
    trees: list[dict],
    *,
    trees_per_iteration: int,
    objective: str,
    base_score: float,
    num_class: int,
    num_feature: int,
    feature_names: list[str],
    feature_types: list[str],
) -> dict:
    """The XGBoost JSON document; ``base_score`` is in output space (a probability
    for logistic objectives), so 0.5 / 1.0 / 0.0 give a zero margin offset for
    logistic / log-link / identity objectives."""
    k = trees_per_iteration
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
                "base_score": f"[{base_score!r}]",
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
