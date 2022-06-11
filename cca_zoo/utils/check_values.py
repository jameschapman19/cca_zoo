import warnings
from typing import Iterable

import numpy as np
from sklearn.utils import check_array


def _check_views(*views: Iterable[np.ndarray], copy=False, accept_sparse=False):
    """

    :param views:
    :param copy:
    """
    if len(views) == 2:
        # This is a bit of a hack to try to match up with the way mvlearn takes views which in turn is a bit of a hack to match up with sklearn.
        # Sklearn expects fit(views,y) so if we want multiview views in sklearn functions we need views to be a list
        if isinstance(views[0], list) and views[1] is None:
            views = views[0]

    n_views = len(views)
    if n_views < 2:
        raise ValueError("Require at least 2 views")

    views = [
        check_array(view, allow_nd=False, copy=copy, accept_sparse=accept_sparse)
        for view in views
    ]

    if not len(set([view.shape[0] for view in views])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    return views


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
            f"number of views passed should match number of parameter {parameter_name}"
            f"len(views)={n_views} and "
            f"len({parameter_name})={len(parameter)}"
        )


def _check_converged_weights(weights, view_index):
    """check the converged weights are not zero."""
    if np.linalg.norm(weights) <= 0:
        warnings.warn(
            f"All result weights are zero in view {view_index}. "
            "Try less regularisation or another initialisation"
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
            "from Parikh 2014 (mu<lam/frobenius(views)**2)."
            "Index of view(s) not meeting the condition: "
            f"{failed_check}."
        )


def _check_batch_size(batch_size, latent_dimensions):
    """check batch size greater than number of latent dimensions and warn user otherwise"""
    if batch_size < latent_dimensions:
        warnings.warn(
            "Objective is unstable when batch size is less than the number of latent dimensions"
        )
