from .simulated import generate_covariance_data, generate_simple_data

try:
    from .utils import CCA_Dataset
except ModuleNotFoundError:
    pass

__all__ = [
    "generate_covariance_data",
    "generate_simple_data",
    "CCA_Dataset",
]
