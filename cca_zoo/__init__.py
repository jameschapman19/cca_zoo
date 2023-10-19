from . import (
    datasets,
    deep,
    linear,
    model_selection,
    visualisation,
    preprocessing,
    sequential,
)

__all__ = [
    "datasets",
    "deep",
    "linear",
    "model_selection",
    "visualisation",
    "preprocessing",
    "sequential",
]
try:
    from . import probabilistic

    __all__.append("probabilistic")
except ModuleNotFoundError:
    pass
