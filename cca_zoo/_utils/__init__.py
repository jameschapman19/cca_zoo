"""Internal utilities for cca-zoo."""

from ._linalg import deflate, gevp, soft_threshold, svd_whiten
from ._validation import perview_parameter, validate_views

__all__ = [
    "validate_views",
    "perview_parameter",
    "svd_whiten",
    "gevp",
    "soft_threshold",
    "deflate",
]
