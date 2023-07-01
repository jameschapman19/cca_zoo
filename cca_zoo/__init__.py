from . import data
from . import model_selection
from . import linear
from . import visualisation
from . import deep

__all__ = [
    "data",
    "model_selection",
    "linear",
    "visualisation",
    "deep",
]
try:
    from . import probabilistic

    __all__.append("probabilistic")
except ModuleNotFoundError:
    pass
