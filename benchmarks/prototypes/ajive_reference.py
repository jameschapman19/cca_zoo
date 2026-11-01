"""Compare AJIVE with the mvlearn implementation on the same data.

mvlearn's AJIVE follows the authors' reference code. It needs an older numpy than this
project, so run its half in a separate environment:

    python ajive_reference.py reference out.npz     # in an environment with mvlearn
    python ajive_reference.py compare out.npz       # in this project's environment

Fixed ranks give the same decomposition up to numerical precision. mvlearn selects the
joint rank by the Wedin bound and the random direction bound; ``AJIVE(n_components="auto")``
uses the random direction bound only.
"""

from __future__ import annotations

import sys
import zlib

import numpy as np

# (n_samples, widths, n_joint, n_individual per view, joint strength)
DESIGNS = {
    "two views": (100, (30, 40), 2, (3, 3), 1.0),
    "three views": (120, (30, 40, 50), 2, (2, 3, 2), 1.0),
    "weak joint": (100, (30, 40), 2, (3, 3), 0.35),
    "no joint": (100, (30, 40), 0, (3, 3), 1.0),
    "one joint": (100, (30, 40), 1, (3, 3), 1.0),
    "three joint": (150, (50, 60), 3, (2, 2), 1.0),
}
SEEDS = range(4)


def data(design: str, seed: int) -> list[np.ndarray]:
    """Views of ``n_joint`` shared factors and individual ones, plus noise."""
    n, widths, n_joint, n_individual, strength = DESIGNS[design]
    rng = np.random.default_rng(zlib.crc32(design.encode()) + seed)
    joint = rng.standard_normal((n, n_joint))
    views = []
    for p, r in zip(widths, n_individual):
        views.append(
            strength * 2 * joint @ rng.standard_normal((n_joint, p)) / np.sqrt(p)
            + 2 * rng.standard_normal((n, r)) @ rng.standard_normal((r, p)) / np.sqrt(p)
            + 0.3 * rng.standard_normal((n, p))
        )
    return views


def reference(path: str) -> None:
    """Run mvlearn: fixed ranks (the true ones) and its own rank selection."""
    from mvlearn.decomposition import AJIVE

    out = {}
    for design, (_, widths, n_joint, n_individual, _) in DESIGNS.items():
        for seed in SEEDS:
            views = data(design, seed)
            key = f"{design}|{seed}"
            signal = [n_joint + r for r in n_individual]
            if n_joint:
                fixed = AJIVE(
                    init_signal_ranks=signal,
                    joint_rank=n_joint,
                    individual_ranks=list(n_individual),
                ).fit(views)
                out[key + "|joint"] = fixed.joint_scores_
                out[key + "|individual"] = (
                    np.stack([np.asarray(m) for m in fixed.individual_mats_])
                    if len(set(widths)) == 1
                    else np.array([0.0])
                )
                out[key + "|individual_list"] = np.array(
                    [np.linalg.norm(m) for m in fixed.individual_mats_]
                )
            selected = AJIVE(
                init_signal_ranks=signal, individual_ranks=list(n_individual)
            ).fit(views)
            out[key + "|auto_rank"] = np.array(selected.joint_rank_)
    np.savez(path, **out)


def compare(path: str) -> None:
    """Compare the joint spaces, the individual parts and the selected ranks."""
    from cca_zoo.linear import AJIVE

    ref = np.load(path)
    print(
        f"{'design':<12} {'joint space (min cos)':>22} {'individual (rel. norm gap)':>28}"
        f" {'rank: mvlearn / ours / true':>30}"
    )
    for design, (_, _, n_joint, n_individual, _) in DESIGNS.items():
        cosines, gaps, theirs, ours = [], [], [], []
        for seed in SEEDS:
            views = data(design, seed)
            key = f"{design}|{seed}"
            signal = [n_joint + r for r in n_individual]
            theirs.append(int(ref[key + "|auto_rank"]))
            try:
                ours.append(
                    AJIVE(
                        n_components="auto",
                        signal_ranks=signal,
                        individual_ranks=list(n_individual),
                        random_state=0,
                    )
                    .fit(views)
                    .n_components_
                )
            except ValueError:
                ours.append(0)  # "no joint structure was found"
            if not n_joint:
                continue
            model = AJIVE(
                n_components=n_joint,
                signal_ranks=signal,
                individual_ranks=list(n_individual),
            ).fit(views)
            a = np.linalg.qr(model.joint_scores_ - model.joint_scores_.mean(0))[0]
            b = np.linalg.qr(ref[key + "|joint"] - ref[key + "|joint"].mean(0))[0]
            cosines.append(np.linalg.svd(a.T @ b, compute_uv=False).min())
            mine = np.array(
                [
                    np.linalg.norm(s @ l.T)
                    for s, l in zip(
                        model.individual_scores_, model.individual_loadings_
                    )
                ]
            )
            gaps.append(
                np.max(
                    np.abs(mine - ref[key + "|individual_list"])
                    / ref[key + "|individual_list"]
                )
            )
        space = f"{np.min(cosines):.6f}" if cosines else "-"
        gap = f"{np.max(gaps):.2e}" if gaps else "-"
        print(
            f"{design:<12} {space:>22} {gap:>28} {str(theirs):>10} / {str(ours):>10} / {n_joint}"
        )


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    reference(path) if mode == "reference" else compare(path)
