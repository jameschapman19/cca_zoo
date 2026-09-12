"""Sparse linear CCA methods, fit by coordinate descent directly on the EY loss."""

from __future__ import annotations

from cca_zoo.sparse._elasticnetcca import ElasticNetCCA

__all__ = ["ElasticNetCCA"]
