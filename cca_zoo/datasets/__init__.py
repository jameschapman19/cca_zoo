"""Multiview dataset utilities: simulated generators and toy real-world loaders."""

from __future__ import annotations

from cca_zoo.datasets._simulated import JointData
from cca_zoo.datasets._toy import load_breast_cancer, load_linnerud

__all__ = [
    "JointData",
    "load_breast_cancer",
    "load_linnerud",
]
