import numpy as np

from cca_zoo.linear._mcca import MCCA, rCCA


def reduce_dims(x):
    U, S, _ = np.linalg.svd(x, full_matrices=False)
    return U @ np.diag(S)


class PLSMixin:
    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"pls": True}


class PLS(rCCA, PLSMixin):
    r"""
    A class used to fit a simple PLS model. This model finds the linear projections of two representations that maximize their covariance.

    Implements PLS by inheriting regularised CCA with maximal regularisation. This is equivalent to solving the following optimization problem:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^Tw_1=1

        w_2^Tw_2=1

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to use, by default 1
    copy_data: bool, optional
        Whether to copy the data, by default True
    random_state: int, optional
        Random state, by default None

    Examples
    --------
    >>> import numpy as np
    >>> X1 = np.random.rand(100, 5)
    >>> X2 = np.random.rand(100, 5)
    >>> pls = PLS(latent_dimensions=2)
    >>> pls.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=1 to enable maximal regularization
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )


class MPLS(MCCA, PLSMixin):
    r"""
    A class used to fit a mutiview PLS model. This model finds the linear projections of two representations that maximize their covariance.

    Implements PLS by inheriting regularised CCA with maximal regularisation. This is equivalent to solving the following optimization problem:

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to use, by default 1
    copy_data: bool, optional
        Whether to copy the data, by default True
    random_state: int, optional
        Random state, by default None
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=1 to enable maximal regularization
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )
