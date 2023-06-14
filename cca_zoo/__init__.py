__all__ = [
    "data",
    "model_selection",
    "classical",
    "visualisation",
]

# if can import deep add to all
try:
    import cca_zoo.deep

    __all__.append("deep")
except ModuleNotFoundError:
    pass
try:
    import cca_zoo.probabilistic

    __all__.append("probabilistic")
except ModuleNotFoundError:
    pass
