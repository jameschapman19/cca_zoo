from cca_zoo.deep.objectives import _MCCALoss
from cca_zoo.linear._gradient._ey import CCA_EY


class CCA_TraceNorm(CCA_EY):
    objective = _MCCALoss()
