import itertools
import warnings
from typing import List, Tuple, Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.manifold import TSNE


def _post_process_cv_results(df):
    cols = [col for col in df.columns if "param_" in col]
    for col in cols:
        df = df.join(
            pd.DataFrame(df[col].tolist()).rename(
                columns=lambda x: col + "_" + str(x + 1)
            )
        ).drop(col, axis=1)
    return df


def cv_plot(cv_results_):
    """
    Plot a hyperparameter surface plot (or multiple surface plots if more than 2 hyperparameters)
    """
    if isinstance(cv_results_, dict):
        cv_results_ = pd.DataFrame(cv_results_)
    cv_results_ = _post_process_cv_results(cv_results_)
    param_cols = [col for col in cv_results_.columns if "param_" in col]
    n_params = len(param_cols)
    n_uniques = [cv_results_[col].nunique() for col in param_cols]
    sub_dfs = []
    sub_scores = []
    if n_params > 4:
        warnings.warn(
            "plot not implemented for more than 4 hyperparameters. Plotting for first 4"
        )
        param_cols = param_cols[:4]
        n_uniques = n_uniques[:4]
    if n_params > 3:
        fig, axs = plt.subplots(
            n_uniques[-2], n_uniques[-1], subplot_kw={"projection": "3d"}
        )
        unique_x = cv_results_[param_cols[-2]].unique()
        unique_y = cv_results_[param_cols[-1]].unique()
        param_pairs = list(itertools.product(unique_x, unique_y))
        for pair in param_pairs:
            mask = (cv_results_[param_cols[-2]] == pair[0]) & (
                cv_results_[param_cols[-1]] == pair[1]
            )
            sub_dfs.append(cv_results_.loc[mask].iloc[:, :-2])
            sub_scores.append(cv_results_[mask].mean_test_score)
    elif n_params == 3:
        fig, axs = plt.subplots(1, n_uniques[-1], subplot_kw={"projection": "3d"})
        unique_x = cv_results_[param_cols[-1]].unique()
        for x in unique_x:
            mask = cv_results_[param_cols[-1]] == x
            sub_dfs.append(cv_results_.loc[mask].iloc[:, :-1])
            sub_scores.append(cv_results_[mask].mean_test_score)
    else:
        sub_dfs.append(cv_results_)
        sub_scores.append(cv_results_.mean_test_score)
        if n_params == 2:
            fig, axs = plt.subplots(1, subplot_kw={"projection": "3d"})
        else:
            fig, axs = plt.subplots(1)
    axs = np.array([axs])
    axs = axs.flatten()
    for i, (ax, sub_df, sub_score) in enumerate(zip(axs, sub_dfs, sub_scores)):
        if len(sub_df.shape) > 1:
            ax.plot_trisurf(
                sub_df[param_cols[0]],
                sub_df[param_cols[1]],
                sub_score,
                cmap=cm.coolwarm,
                linewidth=0.2,
            )
            ax.set_xlabel(param_cols[0])
            ax.set_ylabel(param_cols[1])
            ax.set_zlabel("Score")

    fig.suptitle("Hyperparameter plot")
    return fig


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


def plot_each_view_tsne(
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
        ax.scatter(transformed_views[i][:, 0], transformed_views[i][:, 1])
        ax.set_title(f"Dimension {i + 1}")
    return ax

def plot_explained_variance(
    model, views: Iterable[np.ndarray], ax=None, figsize=None, **kwargs
):
    """
    Plots the explained variance for each dimension

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

    variances = model.explained_variance_(views, **kwargs)
    ax.plot(np.arange(1, variances.shape[-1] + 1), variances.mean(axis=(0, 1)))
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Explained Variance")
    return ax
