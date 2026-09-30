"""Internal utilities for cca-zoo."""

from ._ey import ey_cross_covariance, ey_grad_z, ey_loss
from ._linalg import deflate, gevp, soft_threshold, svd_whiten
from ._validation import perview_parameter, validate_views

__all__ = [
    "deflate",
    "ey_cross_covariance",
    "ey_grad_z",
    "ey_loss",
    "gevp",
    "perview_parameter",
    "soft_threshold",
    "svd_whiten",
    "validate_views",
]
