import warnings

import numpy as np


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
