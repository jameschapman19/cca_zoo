from cca_zoo.deep.objectives import _CCA_GHALoss
from cca_zoo.linear._gradient._ey import CCA_EY


class CCA_GHA(CCA_EY):
    objective = _CCA_GHALoss()

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}
