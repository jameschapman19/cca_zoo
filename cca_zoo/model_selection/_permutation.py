import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm
from cca_zoo.models._cca_base import _CCA_Base


def permutation_test_score(
    estimator: _CCA_Base, X: np.ndarray, Y: np.ndarray, latent_dims: int = 1,
    n_perms: int = 1000, Z: Optional[np.ndarray] = None, W: Optional[np.ndarray] = None,
    sel: Optional[np.ndarray] = None, partial: bool = True,
    parameters: Optional[Dict] = None
            ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutation inference for canonical correlation analysis (CCA) _[1].
    
    This code is adapted from the Matlab function accompagning the paper:
    https://github.com/andersonwinkler/PermCCA/blob/6098d35da79618588b8763c5b4a519438703dba4/permcca.m#L131-L164
    
    Parameters
    ----------
    estimator : _CCA_Base
        The object to use to fit the data. This must be one of the CCA models from
        py:class:`cca_zoo-models` and implementing a `fit` method.
    Y : np.ndarray
        Left set of variables, size N by P.
    X : np.ndarray
        Right set of variables, size N by Q.
    latent_dims : int
        The number of latent dimensions infered during the model fitting. Defaults to
        `1`.
    n_perms : int
        An integer representing the number of permutations. Default is 1000 permutations.
    Z : np.ndarray
        (Optional) Nuisance variables for both (partial CCA) or left
        side (part CCA) only.
    W : np.ndarray
        (Optional) Nuisance variables for the right side only (bipartial CCA).
    sel : np.ndarray
        (Optional) Selection matrix or a selection vector, to use Theil's residuals
        instead of Huh-Jhun's projection. If specified as a vector, it can be made
        of integer indices or logicals. The R unselected rows of Z (S of W) must be full
        rank. Use -1 to randomly select N-R (or N-S) rows.
    partial : bool
        (Optional) Boolean indicating whether this is partial (true) or part (false) CCA.
        Default is true, i.e., partial CCA.
    parameters : dict | None
        (Optional) Any additional keyword arguments required by the given estimator.

    
    Returns
    -------
    p : float
        p-values, FWER corrected via closure.
    r : np.ndarray
        Canonical correlations.
    A : np.ndarray
        Canonical coefficients (X).
    B : np.ndarray
        Canonical coefficients (Y).
    U : np.ndarray
        Canonical variables (X).
    V : np.ndarray
        Canonical variables (Y).
    
    References
    ----------
    .. [1] Winkler AM, Renaud O, Smith SM, Nichols TE. Permutation Inference for
        Canonical Correlation Analysis. NeuroImage. 2020; 117065.

    """

    rng = np.random.RandomState(42)
    lW, cnt  = np.zeros(latent_dims), np.zeros(latent_dims)
    n_obs = X.shape[0]
    if parameters is None:
        parameters = {}

    # Initial fit of the CCA model (without any permutation)
    init_model = estimator(latent_dims=(latent_dims), **parameters)
    init_model.fit((X, Y))

    A, B = init_model.get_loadings((X, Y))
    U, V = init_model.transform((X, Y))

    for i in tqdm(range(n_perms)):

        # If user didn't supply a set of permutations, permute randomly both Y and X.
        # Otherwise, use the permtuation set to shuffle one side only.
        if i == 0:
            # First permutation is no permutation
            X_perm = X
            Y_perm = Y
        else:
            x_idx, y_idx = rng.permutation(n_obs), rng.permutation(n_obs)
            X_perm = X[x_idx]
            Y_perm = Y[y_idx]

        # For each canonical variable
        for k in range(latent_dims):
            
            # Fit the CCA model using the permuted datasets
            perm_model = estimator(latent_dims=(latent_dims - k), **parameters)
            perm_model.fit((X_perm[:, k:], Y_perm[:, k:]))

            # Estimate correlation coefficient for this CCA fit
            r_perm = perm_model.correlations((X_perm[:, k:], Y_perm[:, k:]))[0][1]

            lWtmp = -1 * np.cumsum(np.log(1 - r_perm ** 2)[::-1])[::-1]
            lW[k] = lWtmp[0]

        if i == 0:
            lw1 = lW
        cnt = cnt + (lW >= lw1)
    
    # compute p-values
    p = np.maximum.accumulate(cnt/n_perms)
    
    return p, A, B, U, V
