"""Nonparametric (kernel- and graph-based) CCA methods."""

from ._kcca import KCCA
from ._kgcca import KGCCA
from ._ktcca import KTCCA
from ._manifold_cca import ManifoldCCA

__all__ = ["KCCA", "KGCCA", "KTCCA", "ManifoldCCA"]
