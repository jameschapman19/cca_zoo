from cca_zoo.deep.objectives import _CCA_SVDLoss
from cca_zoo.linear._gradient._ey import CCA_EY, PLS_EY


class CCA_SVD(CCA_EY):
    objective = _CCA_SVDLoss()
