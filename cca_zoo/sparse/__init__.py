"""Sparse linear CCA.

Penalised Eckart-Young models fitted by coordinate descent
(:class:`ElasticNetCCA`, :class:`MultiTaskElasticNetCCA`,
:class:`OrthogonalMatchingPursuitCCA`) and alternating penalised
regressions from the literature (:class:`PMDCCA`, :class:`ADMMCCA`,
:class:`IPLSCCA`, :class:`SpanCCA`, :class:`WaijenborgCCA`,
:class:`ParkhomenkoCCA`, :class:`SAR`).
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
    "ADMMCCA",
    "IPLSCCA",
    "PMDCCA",
    "SAR",
    "ElasticNetCCA",
    "MultiTaskElasticNetCCA",
    "OrthogonalMatchingPursuitCCA",
    "ParkhomenkoCCA",
    "SpanCCA",
    "WaijenborgCCA",
]
