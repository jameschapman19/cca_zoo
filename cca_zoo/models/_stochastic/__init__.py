from ._eigengame import PLSEigenGame, CCAEigenGame, RCCAEigenGame
from ._ghagep import PLSGHAGEP, CCAGHAGEP, RCCAGHAGEP
from ._incrementalpls import IncrementalPLS
from ._stochasticpls import PLSStochasticPower

__all__ = [
    "IncrementalPLS",
    "PLSStochasticPower",
    "PLSGHAGEP",
    "CCAGHAGEP",
    "RCCAGHAGEP",
    "PLSEigenGame",
    "CCAEigenGame",
    "RCCAEigenGame",
]
