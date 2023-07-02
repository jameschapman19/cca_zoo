from . import data, deep, linear, model_selection, visualisation

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
