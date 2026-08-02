"""Internal utilities for cca-zoo."""

from ._ey import ey_cross_covariance, ey_grad_z, ey_loss
from ._linalg import deflate, gevp, soft_threshold, svd_whiten
from ._validation import perview_parameter, validate_views

__all__ = [
    "validate_views",
    "perview_parameter",
    "svd_whiten",
    "gevp",
    "soft_threshold",
    "deflate",
    "ey_cross_covariance",
    "ey_loss",
    "ey_grad_z",
]
