try:
    import cca_zoo.data.deep

    __all__ = ["simulated", "deep"]
except ModuleNotFoundError:
    __all__ = ["simulated"]
