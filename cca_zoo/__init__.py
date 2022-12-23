__all__ = [
    "data",
    "model_selection",
    "models",
    "plotting",]

#if can import deepmodels add to all
try:
    import cca_zoo.deepmodels
    __all__.append("deepmodels")
except ModuleNotFoundError:
    pass
try:
    import cca_zoo.probabilisticmodels
    __all__.append("probabilisticmodels")
except ModuleNotFoundError:
    pass
