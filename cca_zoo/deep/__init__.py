"""Deep multiview models in PyTorch Lightning; requires the ``deep`` extra."""

from __future__ import annotations

import importlib.util

_torch_available = importlib.util.find_spec("torch") is not None
_lightning_available = importlib.util.find_spec("lightning") is not None

if _torch_available and _lightning_available:
    from cca_zoo.deep import objectives
    from cca_zoo.deep._barlowtwins import BarlowTwins
    from cca_zoo.deep._base import BaseDeep
    from cca_zoo.deep._data import MultiviewDataset
    from cca_zoo.deep._dcca import DCCA
    from cca_zoo.deep._dcca_ey import DCCAEY
    from cca_zoo.deep._dcca_noi import DCCANOI
    from cca_zoo.deep._dcca_sdl import DCCASDL
    from cca_zoo.deep._dccae import DCCAE
    from cca_zoo.deep._dgcca import DGCCA
    from cca_zoo.deep._dmcca import DMCCA
    from cca_zoo.deep._dpcca import DPCCA
    from cca_zoo.deep._dtcca import DTCCA
    from cca_zoo.deep._dvcca import DVCCA, DVCCAPrivate
    from cca_zoo.deep._lejepa import LeJEPA
    from cca_zoo.deep._nrdcca import NRDCCA
    from cca_zoo.deep._splitae import SplitAE
    from cca_zoo.deep._vicreg import VICReg

    __all__ = [
        "DCCA",
        "DCCAE",
        "DCCAEY",
        "DCCANOI",
        "DCCASDL",
        "DGCCA",
        "DMCCA",
        "DPCCA",
        "DTCCA",
        "DVCCA",
        "NRDCCA",
        "BarlowTwins",
        "BaseDeep",
        "DVCCAPrivate",
        "LeJEPA",
        "MultiviewDataset",
        "SplitAE",
        "VICReg",
        "objectives",
    ]
else:
    __all__ = []

    def __getattr__(name: str) -> object:
        if name in {
            "DCCA",
            "DCCAE",
            "DCCAEY",
            "DCCANOI",
            "DCCASDL",
            "DGCCA",
            "DMCCA",
            "DPCCA",
            "DTCCA",
            "DVCCA",
            "NRDCCA",
            "BarlowTwins",
            "BaseDeep",
            "DVCCAPrivate",
            "LeJEPA",
            "MultiviewDataset",
            "SplitAE",
            "VICReg",
            "objectives",
        }:
            raise ImportError(
                f"{name} requires the deep extra: pip install 'cca-zoo[deep]'."
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
