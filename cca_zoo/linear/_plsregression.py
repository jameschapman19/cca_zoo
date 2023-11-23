from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from cca_zoo.linear import PLS


class PLSRegression(PLS):
    @property
    def coef_(self):
        return self.weights_[0] @ self.weights_[1].T

    def predict(self, X):
        check_is_fitted(self)
        check_array(
            X,
            copy=True,
            accept_sparse=False,
            accept_large_sparse=False,
        )
        Y_pred = X @ self.coef_
        return Y_pred
