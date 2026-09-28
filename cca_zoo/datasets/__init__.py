"""Simulated and toy multiview datasets."""

from __future__ import annotations

from cca_zoo.datasets._simulated import make_joint_data
from cca_zoo.datasets._toy import load_breast_cancer, load_linnerud

__all__ = [
    "load_breast_cancer",
    "load_linnerud",
    "make_joint_data",
]
