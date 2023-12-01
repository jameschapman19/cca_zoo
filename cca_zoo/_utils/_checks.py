import warnings

import numpy as np
from sklearn.utils import check_array


def check_Xs(
    Xs,
    multiview=False,
    enforce_views=None,
    copy=False,
    return_dimensions=False,
):
    # Authors: Pierre Ablin
    # Copyright (c) 2020 The mvlearn developers.
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    r"""
    Checks Xs and ensures it to be a list of 2D matrices.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, (default=False)
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs,
        and operations on the output will not affect
        the input.
        If False, the returned Xs is a view of the input Xs,
        and operations on the output will change the input.

    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of views and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each view.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    n_views : int
        The number of views in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_features : list
        List of length ``n_views`` containing the number of features in
        each view. Returned only if ``return_dimensions`` is ``True``.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_views = len(Xs)
    if n_views == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if n_views == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and n_views != enforce_views:
            msg = "Wrong number of views. Expected {} but found {}".format(
                enforce_views, n_views
            )
            raise ValueError(msg)

    Xs = [check_array(X, allow_nd=False, copy=copy) for X in Xs]

    if not len(set([X.shape[0] for X in Xs])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_views, n_samples, n_features
    else:
        return Xs


def _process_parameter(parameter_name: str, parameter, default, n_views: int):
    if parameter is None:
        parameter = [default] * n_views
    elif not isinstance(parameter, (list, tuple)):
        parameter = [parameter] * n_views
    _check_parameter_number(parameter_name, parameter, n_views)
    return parameter


def _check_parameter_number(parameter_name: str, parameter, n_views: int):
    if len(parameter) != n_views:
        raise ValueError(
            f"number of representations passed should match number of parameter {parameter_name}"
            f"len(representations)={n_views} and "
            f"len({parameter_name})={len(parameter)}"
        )


def _check_Parikh2014(mus, lams, views):
    """Return index of the view which the data not matching the condition
    documented in Parikh 2014."""
    failed_check = [
        i
        for i, (mu, lam, view) in enumerate(zip(mus, lams, views))
        if mu < lam / np.linalg.norm(view) ** 2
    ]
    if failed_check:
        raise ValueError(
            "mu, lam, view not matching condition specified "
            "from Parikh 2014 (mu<lam/frobenius(representations)**2)."
            "Index of view(s) not meeting the condition: "
            f"{failed_check}."
        )


def _check_batch_size(batch_size, latent_dimensions):
    """check batch size greater than number of latent dimensions and warn user otherwise"""
    if batch_size < latent_dimensions:
        warnings.warn(
            "Objective is unstable when batch size is less than the number of latent dimensions"
        )


def check_tsne_support(caller_name):
    try:
        import openTSNE  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires openTSNE. "
            "Please install openTSNE using `pip install openTSNE`"
        )


def check_umap_support(caller_name):
    try:
        import umap  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires umap. "
            "Please install umap using `pip install umap-learn`"
        )


def check_seaborn_support(caller_name):
    try:
        import seaborn  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires seaborn. "
            "Please install seaborn using `pip install seaborn`"
        )


def check_arviz_support(caller_name):
    try:
        import arviz as az  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires arviz. "
            "Please install arviz using `pip install arviz`"
        )


def check_gglasso_support(caller_name):
    try:
        import gglasso  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires gglasso. "
            "Please install glasso using `pip install gglasso`"
        )


def check_graphviz_support(caller_name):
    try:
        import graphviz  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires rdata. "
            "Please install pyreadr using `pip install rdata`"
        )


def check_rdata_support(caller_name):
    try:
        import rdata  # noqa
    except ImportError:
        raise ImportError(
            f"{caller_name} requires rdata. "
            "Please install pyreadr using `pip install rdata`"
        )
