"""Sparse linear CCA methods, fit by coordinate descent directly on the EY loss."""

from __future__ import annotations

from cca_zoo.sparse._elasticnetcca import ElasticNetCCA
from cca_zoo.sparse._multitaskelasticnetcca import MultiTaskElasticNetCCA
from cca_zoo.sparse._ompcca import OrthogonalMatchingPursuitCCA

__all__ = ["ElasticNetCCA", "MultiTaskElasticNetCCA", "OrthogonalMatchingPursuitCCA"]
