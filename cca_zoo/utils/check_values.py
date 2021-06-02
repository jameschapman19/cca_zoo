import numpy as np


def _check_parameter_number(c, n_views):
    if len(c) != n_views:
        raise ('number of views passed shout match number of parameter c '
                f'for regularisation. len(views)={n_views} and '
                f'len(c)={len(c)}')


def _check_converged_weights(weights, view_index):
    """check the converged weights are not zero."""
    if np.linalg.norm(weights) <= 0:
        raise ValueError(f"All result weights are zero in view {view_index}. "
                         "Try less regularisation or another initialisation")


def _check_Parikh2014(mus, lams, views):
    """Return index of the view which the data not matching the condition
    documented in Parikh 2014."""
    failed_check =  [
        i
        for i, (mu, lam, view) in enumerate(zip(mus, lams, views))
        if mu <= lam / np.linalg.norm(view) ** 2
    ]
    if failed_check:
        raise ValueError("mu, lam, view not matching condition specified "
                         "from Parikh 2014 (mu<lam/frobenius(X)**2)."
                         "Index of view(s) not meeting the condition: "
                         f"{failed_check}.")