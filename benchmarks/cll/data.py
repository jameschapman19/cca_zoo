"""The CLL multi-omics cohort of Argelaguet et al. (2018), from the MOFAdata package.

200 chronic lymphocytic leukaemia patients with drug response, methylation, mRNA
and somatic mutations. Drugs, methylation and mRNA are each missing for whole
patients, which is the case ``IncompleteGCCA`` is built for; the mutation view has
scattered missing entries, which are filled with the column mean.

IGHV mutation status and trisomy 12, the two best-established axes of CLL biology,
are columns of the mutation view. They are taken out of it, to be used as labels
that no method sees.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rdata

_URL = "https://raw.githubusercontent.com/bioFAM/MOFAdata/master/data/CLL_data.RData"
CACHE = Path(__file__).parent / "cache"
LABELS = ("IGHV", "trisomy12")
MAX_MISSING_MUTATION = 0.5  # drop mutation features missing in more samples than this


@dataclass(frozen=True)
class CLL:
    """The standardised views, NaN where a patient lacks a view, and the labels."""

    names: list[str]
    views: list[np.ndarray]  # (n_samples, n_features_i), standardised on observed rows
    observed: np.ndarray  # (n_samples, n_views) bool
    labels: dict[str, np.ndarray]  # 0/1, NaN where unknown
    samples: list[str]


def load() -> CLL:
    """Download (once) and prepare the cohort."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "CLL_data.RData"
    if not path.exists():
        urllib.request.urlretrieve(_URL, path)
    raw = rdata.read_rda(path)["CLL_data"]
    samples = [str(s) for s in raw["Drugs"].coords["dim_1"].values]

    mutations = raw["Mutations"]
    features = [str(f) for f in mutations.coords["dim_0"].values]
    table = mutations.values.astype(float)  # features x samples
    labels = {name: table[features.index(name)].copy() for name in LABELS}
    keep = [
        j
        for j, f in enumerate(features)
        if f not in LABELS and np.isnan(table[j]).mean() <= MAX_MISSING_MUTATION
    ]

    names, views = [], []
    for name in ("Drugs", "Methylation", "mRNA", "Mutations"):
        values = raw[name].values.astype(float).T  # samples x features
        if name == "Mutations":
            values = values[:, keep]
            values = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
        seen = ~np.isnan(values).all(axis=1)
        mean, sd = values[seen].mean(axis=0), values[seen].std(axis=0)
        values = (values - mean) / np.where(sd > 0, sd, 1.0)
        names.append(name)
        views.append(values)
    observed = np.stack([~np.isnan(v).all(axis=1) for v in views], axis=1)
    return CLL(names, views, observed, labels, samples)
