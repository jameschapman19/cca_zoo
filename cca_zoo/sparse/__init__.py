"""Sparse linear CCA methods.

Two mechanism families live here:

- EY-loss coordinate descent (:class:`ElasticNetCCA`,
  :class:`MultiTaskElasticNetCCA`, :class:`OrthogonalMatchingPursuitCCA`) --
  penalised extensions of the same unconstrained Eckart-Young objective
  :class:`~cca_zoo.linear.gradient.CCAEY` uses.
- Alternating Least Squares (:class:`PMDCCA`, :class:`ADMMCCA`,
  :class:`IPLSCCA`, :class:`SpanCCA`, :class:`WaijenborgCCA`,
  :class:`ParkhomenkoCCA`, :class:`SAR`) -- each a from-the-literature sparse
  CCA algorithm with its own penalty and fitting loop; see
  :mod:`cca_zoo.sparse._iterative`'s module docstring for the shared ALS
  convention they follow.
"""

from __future__ import annotations

from cca_zoo.sparse._elasticnetcca import ElasticNetCCA
from cca_zoo.sparse._iterative import (
    ADMMCCA,
    IPLSCCA,
    PMDCCA,
    SAR,
    ParkhomenkoCCA,
    SpanCCA,
    WaijenborgCCA,
)
from cca_zoo.sparse._multitaskelasticnetcca import MultiTaskElasticNetCCA
from cca_zoo.sparse._ompcca import OrthogonalMatchingPursuitCCA

__all__ = [
    "ElasticNetCCA",
    "MultiTaskElasticNetCCA",
    "OrthogonalMatchingPursuitCCA",
    "PMDCCA",
    "ADMMCCA",
    "IPLSCCA",
    "SpanCCA",
    "WaijenborgCCA",
    "ParkhomenkoCCA",
    "SAR",
]
