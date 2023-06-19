import numpy as np
class Deflation:
    def __init__(self, X, Y=None):
        self.X=X
        self.Y=Y
        self._Cxy=self.C[:self.X.shape[1], self.X.shape[1]:]

    @property
    def C(self):
        if self.Y is None:
            self.Y = self.X
        self._C=np.cov(self.X, self.Y, rowvar=False)
        return self._C


class ProjectionDeflation(Deflation):

    def deflate(self,x_weights, y_weights=None):
        if y_weights is None:
            y_weights=x_weights
        self.X-=self.X@np.outer(x_weights,x_weights)
        self.Y-=self.Y@np.outer(y_weights,y_weights)

    @property
    def Cxy(self):
        self._Cxy = self.C[:self.X.shape[1], self.X.shape[1]:]
        return self._Cxy

class GeneralizedDeflation(Deflation):

    def __init__(self, views: np.ndarray):
        super().__init__(views)
        self.Bx=np.eye(self.Cxx.shape[0])
        self.By=np.eye(self.Cyy.shape[0])

    @property
    def Cxy(self):
        return self._Cxy

    def deflate(self,x_weights, y_weights=None):
        if y_weights is None:
            y_weights=x_weights
        qx=self.Bx@x_weights
        qy=self.By@y_weights
        self._Cxy = (np.eye(self.Cxx.shape[0]) - np.outer(qx,qx)) @ self._Cxy @ (np.eye(self.Cyy.shape[0]) - np.outer(qy,qy))
        self.Bx-=self.Bx@np.outer(qx,qx)
        self.By-=self.By@np.outer(qy,qy)

class DeflationMixin:
    """Mixin class for deflation methods."""

    def __init__(self, deflation: str = "projection"):
        """Initialize the deflation method.

        Args:
            deflation (str, optional): Deflation method to use. Defaults to "projection".
        """
        if deflation=='projection':
            self.deflation_method=ProjectionDeflation
        elif deflation=='generalized':
            self.deflation_method=GeneralizedDeflation
        else:
            raise ValueError(f"Invalid deflation method: {deflation}. "
                             f"Must be one of ['hotelling', 'projection', 'generalized'].")

    def deflate(self, views: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # if no attribute self.deflation then initialize it
        if not hasattr(self, "deflation"):
            self.deflation= self.deflation_method(views)
        return self.deflation.deflate(weights)