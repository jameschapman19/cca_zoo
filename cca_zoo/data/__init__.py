from . import simulated

try:
    from . import deep

    __all__ = [
        "simulated",
        "deep"
    ]
except ModuleNotFoundError:
    __all__ = [
        "simulated"
    ]
