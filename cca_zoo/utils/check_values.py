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
        import openTSNE
    except ImportError:
        raise ImportError(
            f"{caller_name} requires openTSNE. "
            "Please install openTSNE using `pip install openTSNE`"
        )


def check_umap_support(caller_name):
    try:
        import umap
    except ImportError:
        raise ImportError(
            f"{caller_name} requires umap. "
            "Please install umap using `pip install umap-learn`"
        )


def check_seaborn_support(caller_name):
    try:
        import seaborn
    except ImportError:
        raise ImportError(
            f"{caller_name} requires seaborn. "
            "Please install seaborn using `pip install seaborn`"
        )


def check_arviz_support(caller_name):
    try:
        import arviz as az
    except ImportError:
        raise ImportError(
            f"{caller_name} requires arviz. "
            "Please install arviz using `pip install arviz`"
        )


def check_gglasso_support(caller_name):
    try:
        import gglasso
    except ImportError:
        raise ImportError(
            f"{caller_name} requires gglasso. "
            "Please install glasso using `pip install gglasso`"
        )


def check_rdata_support(caller_name):
    try:
        import rdata
    except ImportError:
        raise ImportError(
            f"{caller_name} requires rdata. "
            "Please install pyreadr using `pip install rdata`"
        )
