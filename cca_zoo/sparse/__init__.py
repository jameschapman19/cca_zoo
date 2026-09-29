"""Sparse linear CCA.

Penalised Eckart-Young models fitted by coordinate descent
(:class:`ElasticNetCCA`, :class:`MultiTaskElasticNetCCA`,
:class:`OrthogonalMatchingPursuitCCA`) and alternating penalised
regressions from the literature (:class:`PMDCCA`, :class:`ADMMCCA`,
:class:`IPLSCCA`, :class:`SpanCCA`, :class:`WaijenborgCCA`,
:class:`ParkhomenkoCCA`, :class:`SAR`).
"""

from __future__ import annotations

from cca_zoo.sparse._admm import ADMMCCA
from cca_zoo.sparse._elasticnetcca import ElasticNetCCA
from cca_zoo.sparse._ipls import IPLSCCA
from cca_zoo.sparse._multitaskelasticnetcca import MultiTaskElasticNetCCA
from cca_zoo.sparse._ompcca import OrthogonalMatchingPursuitCCA
from cca_zoo.sparse._parkhomenko import ParkhomenkoCCA
from cca_zoo.sparse._pmd import PMDCCA
from cca_zoo.sparse._sar import SAR
from cca_zoo.sparse._span import SpanCCA
from cca_zoo.sparse._waijenborg import WaijenborgCCA

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
