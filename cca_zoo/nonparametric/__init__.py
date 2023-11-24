"""
Nonparametric CCA methods
"""
from ._kcca import KCCA, KGCCA, KTCCA, KMCCA
from ._ncca import NCCA

__all__ = ["KCCA", "KMCCA", "KGCCA", "KTCCA", "NCCA"]
