from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pairwise_correlations(
    model,
    views: Iterable[np.ndarray],
    ax=None,
    figsize=None,
    **kwargs,
):
    """
    Plots the pairwise correlations between the views in each dimension

    Parameters
    ----------
    views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
    ax : matplotlib axes object, optional
        If not provided, a new figure will be created.
    figsize : tuple, optional
        The size of the figure to create. If not provided, the default matplotlib figure size will be used.
    kwargs : any additional keyword arguments required by the given model

    Returns
    -------
    ax : matplotlib axes object

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    corrs = model.pairwise_correlations(views, **kwargs)
    for i in range(corrs.shape[-1]):
        sns.heatmap(
            corrs[:, :, i],
            annot=True,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="RdBu_r",
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"Dimension {i + 1}")
    return ax


def plot_pairwise_scatter(
    model, views: Iterable[np.ndarray], ax=None, figsize=None, **kwargs
):
    """
    Plots the pairwise scatterplots between the views in each dimension

    Parameters
    ----------
    views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
    ax : matplotlib axes object, optional
        If not provided, a new figure will be created.
    figsize : tuple, optional
        The size of the figure to create. If not provided, the default matplotlib figure size will be used.
    kwargs : any additional keyword arguments required by the given model

    Returns
    -------
    ax : matplotlib axes object

    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    transformed_views = model.transform(views, **kwargs)
    for i in range(len(transformed_views)):
        for j in range(i + 1, len(transformed_views)):
            ax.scatter(transformed_views[i], transformed_views[j])
            ax.set_title(f"Dimension {i + 1} vs Dimension {j + 1}")
    return ax
