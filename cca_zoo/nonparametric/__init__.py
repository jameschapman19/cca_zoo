"""Nonparametric (kernel-based) CCA methods."""

from ._kcca import KCCA
from ._kgcca import KGCCA
from ._ktcca import KTCCA

__all__ = ["KCCA", "KGCCA", "KTCCA"]
