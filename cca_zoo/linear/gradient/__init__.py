"""Linear models fitted by minimising the Eckart-Young loss."""

from cca_zoo.linear.gradient._cca_ey import CCAEY
from cca_zoo.linear.gradient._huber_cca import HuberCCA
from cca_zoo.linear.gradient._pls_ey import PLSEY

__all__ = ["CCAEY", "PLSEY", "HuberCCA"]
