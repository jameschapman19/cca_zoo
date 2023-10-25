from cca_zoo.deep.objectives import CCA_SVDLoss
from cca_zoo.linear._gradient._ey import CCA_EY


#
#
class CCA_SVD(CCA_EY):
    objective = CCA_SVDLoss()


#
#
# class PLS_SVD(CCA_EY):
#     objective = PLS_SVDLoss()
