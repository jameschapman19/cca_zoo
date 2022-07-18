from .simulated import linear_simulated_data, simple_simulated_data

try:
    from .utils import CCA_Dataset

    __all__ = [
        "linear_simulated_data",
        "simple_simulated_data",
        "CCA_Dataset",
    ]
except ModuleNotFoundError:
    __all__ = [
        "linear_simulated_data",
        "simple_simulated_data",
    ]
