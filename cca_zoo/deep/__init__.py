"""Deep multiview CCA models powered by PyTorch Lightning.

This module is only available when both ``torch`` and ``lightning``
are installed.  Import errors are deferred to usage time rather than
raised at import of ``cca_zoo``.
"""

from __future__ import annotations

import importlib.util

_torch_available = importlib.util.find_spec("torch") is not None
_lightning_available = importlib.util.find_spec("lightning") is not None

if _torch_available and _lightning_available:
    from cca_zoo.deep import objectives
    from cca_zoo.deep._barlowtwins import BarlowTwins
    from cca_zoo.deep._base import BaseDeep
    from cca_zoo.deep._dcca import DCCA
    from cca_zoo.deep._dcca_ey import DCCA_EY
    from cca_zoo.deep._dcca_noi import DCCA_NOI
    from cca_zoo.deep._dcca_sdl import DCCA_SDL
    from cca_zoo.deep._dccae import DCCAE
    from cca_zoo.deep._dtcca import DTCCA
    from cca_zoo.deep._dvcca import DVCCA
    from cca_zoo.deep._splitae import SplitAE
    from cca_zoo.deep._vicreg import VICReg

    __all__ = [
        "BaseDeep",
        "BarlowTwins",
        "DCCA",
        "DCCA_EY",
        "DCCA_NOI",
        "DCCA_SDL",
        "DCCAE",
        "DTCCA",
        "DVCCA",
        "SplitAE",
        "VICReg",
        "objectives",
    ]
else:
    __all__ = []
