"""Deprecated re-exports for classes that moved to another top-level module.

Kept importable from ``cca_zoo.linear`` for backward compatibility, but
intentionally left out of ``cca_zoo.linear``'s ``__all__`` (and therefore its
docs) since they are being removed from here in a future release -- import
them from their new module instead.
"""

from __future__ import annotations

from sklearn.utils import deprecated

from cca_zoo.sparse import SAR as _SAR
from cca_zoo.sparse import SCCAADMM as _SCCAADMM
from cca_zoo.sparse import SCCAIPLS as _SCCAIPLS
from cca_zoo.sparse import SCCAPMD as _SCCAPMD
from cca_zoo.sparse import ParkhomenkoCCA as _ParkhomenkoCCA
from cca_zoo.sparse import SCCASpan as _SCCASpan
from cca_zoo.sparse import WaijenborgCCA as _WaijenborgCCA
from cca_zoo.stochastic import StochasticCCAEY as _StochasticCCAEY


@deprecated("Moved to cca_zoo.sparse; import SCCAPMD from there instead.")
class SCCAPMD(_SCCAPMD):
    pass


@deprecated("Moved to cca_zoo.sparse; import SCCAADMM from there instead.")
class SCCAADMM(_SCCAADMM):
    pass


@deprecated("Moved to cca_zoo.sparse; import SCCAIPLS from there instead.")
class SCCAIPLS(_SCCAIPLS):
    pass


@deprecated("Moved to cca_zoo.sparse; import SCCASpan from there instead.")
class SCCASpan(_SCCASpan):
    pass


@deprecated("Moved to cca_zoo.sparse; import WaijenborgCCA from there instead.")
class WaijenborgCCA(_WaijenborgCCA):
    pass


@deprecated("Moved to cca_zoo.sparse; import ParkhomenkoCCA from there instead.")
class ParkhomenkoCCA(_ParkhomenkoCCA):
    pass


@deprecated("Moved to cca_zoo.sparse; import SAR from there instead.")
class SAR(_SAR):
    pass


@deprecated("Moved to cca_zoo.stochastic; import StochasticCCAEY from there instead.")
class StochasticCCAEY(_StochasticCCAEY):
    pass
