"""Train one tree per library on the same data and draw the native structures side by side."""

import json
import sys
import tempfile
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tree_structures.png")
COLORS = {"XGBoost": "#2a78d6", "LightGBM": "#eb6834", "CatBoost": "#1baf7a"}  # slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

rng = np.random.default_rng(3)
n = 4000
X = rng.normal(size=(n, 3))
X[rng.random(n) < 0.05, 1] = np.nan
y = (
    5
    + 2 * (X[:, 0] > 0)
    + 1.5 * (np.nan_to_num(X[:, 1], nan=-1) > 0.3) * (X[:, 0] < 0.8)
    + 0.8 * X[:, 2]
    + rng.normal(scale=0.4, size=n)
)
LR = 0.3

# --- native trees as nested (label, note, left, right) / ("leaf", value) -------


def xgboost_tree():
    bst = xgb.train(
        {"max_depth": 3, "eta": LR, "tree_method": "hist"}, xgb.DMatrix(X, y), 1
    )
    t = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"][0]

    def build(i):
        if t["left_children"][i] == -1:
            return {"leaf": t["split_conditions"][i]}
        return {
            "split": f"f{t['split_indices'][i]} < {t['split_conditions'][i]:.3g}",
            "nan": "NaN→" + ("L" if t["default_left"][i] else "R"),
            "left": build(t["left_children"][i]),
            "right": build(t["right_children"][i]),
        }

    return build(0)


def lightgbm_tree():
    b = lgb.train(
        {"objective": "regression", "num_leaves": 8, "learning_rate": LR, "verbose": -1},
        lgb.Dataset(X, y),
        1,
    )

    def build(node):
        if "leaf_value" in node:
            return {"leaf": node["leaf_value"]}
        return {
            "split": f"f{node['split_feature']} ≤ {node['threshold']:.3g}",
            "nan": "NaN→" + ("L" if node["default_left"] else "R"),
            "left": build(node["left_child"]),
            "right": build(node["right_child"]),
        }

    return build(b.dump_model()["tree_info"][0]["tree_structure"])


def catboost_tree():
    m = cb.CatBoostRegressor(
        iterations=1, depth=3, learning_rate=LR, verbose=0, allow_writing_files=False
    ).fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        m.save_model(Path(d) / "m.json", format="json")
        doc = json.loads((Path(d) / "m.json").read_text())
    tree = doc["oblivious_trees"][0]
    flat = [f["flat_feature_index"] for f in doc["features_info"]["float_features"]]
    splits, leaves = tree["splits"], tree["leaf_values"]

    def build(level, bits):
        if level == len(splits):
            return {"leaf": leaves[bits]}
        bit = len(splits) - 1 - level
        s = splits[bit]
        return {
            "split": f"f{flat[s['float_feature_index']]} > {s['border']:.2f}",
            "nan": "NaN→L",
            "left": build(level + 1, bits),
            "right": build(level + 1, bits | 1 << bit),
        }

    return build(0, 0), doc["scale_and_bias"]


# --- drawing ------------------------------------------------------------------


def layout(node, depth=0, leaves=None, pos=None):
    """Assign (x, depth) with leaves spaced evenly in left-to-right order."""
    leaves = [0] if leaves is None else leaves
    pos = {} if pos is None else pos
    if "leaf" in node:
        x = leaves[0]
        leaves[0] += 1
    else:
        xl = layout(node["left"], depth + 1, leaves, pos)
        xr = layout(node["right"], depth + 1, leaves, pos)
        x = (xl + xr) / 2
    pos[id(node)] = (x, depth)
    return x


def depth_of(node):
    return 0 if "leaf" in node else 1 + max(depth_of(node["left"]), depth_of(node["right"]))


def draw(ax, root, color, max_depth, max_leaves):
    pos = {}
    layout(root, pos=pos)
    n_leaves = sum(1 for _ in _leaves(root))

    def walk(node):
        x, d = pos[id(node)]
        y = -d
        if "leaf" not in node:
            for side, tag in (("left", "yes" if False else ""), ("right", "")):
                cx, cd = pos[id(node[side])]
                ax.plot([x, cx], [y - 0.14, -cd + 0.14], color=GRID, lw=1.5, zorder=1)
            ax.text(
                x, y, node["split"], ha="center", va="center", fontsize=8.5, color=INK, zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.4),
            )
            ax.text(x, y - 0.24, node["nan"], ha="center", va="top", fontsize=7, color=MUTED)
            walk(node["left"])
            walk(node["right"])
        else:
            ax.text(
                x, y, f"{node['leaf']:.2f}", ha="center", va="center", fontsize=8.5, color=INK,
                zorder=3, bbox=dict(boxstyle="round,pad=0.25", fc=color + "33", ec=color, lw=1.4),
            )

    walk(root)
    ax.set_xlim(-0.7, max_leaves - 0.3)
    ax.set_ylim(-max_depth - 0.55, 0.5)
    ax.axis("off")
    return n_leaves


def _leaves(node):
    if "leaf" in node:
        yield node
    else:
        yield from _leaves(node["left"])
        yield from _leaves(node["right"])


xgb_t, lgb_t = xgboost_tree(), lightgbm_tree()
cb_t, (cb_scale, cb_bias) = catboost_tree()
trees = {"XGBoost": xgb_t, "LightGBM": lgb_t, "CatBoost": cb_t}
max_depth = max(depth_of(t) for t in trees.values())
max_leaves = max(len(list(_leaves(t))) for t in trees.values())

notes = {
    "XGBoost": [
        "grows level by level (here capped at depth 3)",
        "left if x < threshold (strict)",
        "NaN follows a learned default direction",
        "leaf = shrunken correction; base_score kept separate",
    ],
    "LightGBM": [
        "grows leaf by leaf: lopsided, depth is uncapped",
        "left if x ≤ threshold",
        "NaN follows a learned default direction",
        "leaf already includes the initial mean",
    ],
    "CatBoost": [
        "symmetric: one split per level, always 2^d leaves",
        "right if x > border",
        "NaN compares as lowest value (goes left)",
        f"leaf = correction; model adds bias {cb_bias[0]:.2f}",
    ],
}

fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.6))
fig.patch.set_facecolor("#fcfcfb")
for ax, (name, tree) in zip(axes, trees.items()):
    leaves = draw(ax, tree, COLORS[name], max_depth, max_leaves)
    ax.set_title(
        f"{name}  ·  {leaves} leaves, depth {depth_of(tree)}", fontsize=13, color=INK,
        loc="left", fontweight="bold", pad=10,
    )
    ax.plot([0, 1], [1.2, 1.2], color=COLORS[name], lw=4, transform=ax.transAxes, clip_on=False)
    for i, line in enumerate(notes[name]):
        ax.text(0.0, -0.03 - 0.055 * i, "•  " + line, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, ha="left", va="top")
fig.suptitle("One tree, same data: native structure per library", x=0.01, ha="left",
             fontsize=15, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
print(OUT, {k: (len(list(_leaves(v))), depth_of(v)) for k, v in trees.items()})
