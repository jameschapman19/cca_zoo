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
    ParkhomenkoCCA,
    SpanCCA,
    WaijenborgCCA,
)
from cca_zoo.sparse._iterative import SAR as SAR
from cca_zoo.sparse._iterative import SCCA_ADMM as SCCA_ADMM
from cca_zoo.sparse._iterative import SCCA_IPLS as SCCA_IPLS
from cca_zoo.sparse._iterative import SCCA_PMD as SCCA_PMD
from cca_zoo.sparse._iterative import SCCAADMM as SCCAADMM
from cca_zoo.sparse._iterative import SCCAIPLS as SCCAIPLS
from cca_zoo.sparse._iterative import SCCAPMD as SCCAPMD
from cca_zoo.sparse._iterative import ElasticCCA as ElasticCCA
from cca_zoo.sparse._iterative import SCCA_Span as SCCA_Span
from cca_zoo.sparse._iterative import SCCASpan as SCCASpan
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
# Deprecated aliases (SCCAPMD, SCCAADMM, SCCAIPLS, SCCASpan -- the SCCA
# prefix is redundant with this module's own name -- SCCA_PMD, SCCA_ADMM,
# SCCA_IPLS, SCCA_Span -- underscored renames -- and ElasticCCA, renamed
# WaijenborgCCA) stay importable here for backward compatibility but are
# intentionally left out of __all__/docs, since they are being removed in a
# future release.
