"""Stochastic (mini-batch) CCA methods.

Currently a single class, but split out from :mod:`cca_zoo.linear` as its
own module rather than folded into the full-batch EY-loss classes there:
mini-batch fitting is a genuinely different operational regime (streaming
or out-of-core data) from every other class in this package, which all
assume the full dataset fits in memory for a single fit call.
"""

from __future__ import annotations

from cca_zoo.stochastic._stochastic_cca_ey import StochasticCCAEY

__all__ = ["StochasticCCAEY"]
