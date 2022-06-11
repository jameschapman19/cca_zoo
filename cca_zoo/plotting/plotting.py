import itertools
import warnings
from typing import Union, Tuple, List

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


def pairplot_train_test(
    train_scores: Union[Tuple[np.ndarray], List[np.ndarray]],
    test_scores: Union[Tuple[np.ndarray], List[np.ndarray]] = None,
    title="",
):
    """
    Makes a pair plot showing the projections of each view against each other for each dimensions. Coloured by train and test

    :param train_scores: projections of training data which can be accessed by model.scores
    :param test_scores: projections of test data obtained by model.transform(*test_data)
    :param title: Figure title
    """
    data = pd.DataFrame(
        {"phase": np.asarray(["train"] * train_scores[0].shape[0]).astype(str)}
    )
    x_vars = [f"view 1 projection {f + 1}" for f in range(train_scores[0].shape[1])]
    y_vars = [f"view 2 projection {f + 1}" for f in range(train_scores[1].shape[1])]
    data[x_vars] = train_scores[0]
    data[y_vars] = train_scores[1]
    if test_scores is not None:
        test_data = pd.DataFrame(
            {"phase": np.asarray(["test"] * test_scores[0].shape[0]).astype(str)}
        )
        test_data[x_vars] = test_scores[0]
        test_data[y_vars] = test_scores[1]
        data = pd.concat([data, test_data], axis=0)
    cca_pp = sns.pairplot(data, hue="phase", x_vars=x_vars, y_vars=y_vars)
    cca_pp.fig.suptitle(title)
    return cca_pp


def pairplot_label(
    scores: Union[Tuple[np.ndarray], List[np.ndarray]],
    labels=None,
    label_name=None,
    title="",
):
    """
    Makes a pair plot showing the projections of each view against each other for each dimensions. Coloured by categorical label

    :param scores: projections of data obtained by model.transform(*data)
    :param labels: array of labels
    :param label_name: name of label for legend
    :param title: Figure title
    """
    if label_name is None:
        label_name = "label"
    data = pd.DataFrame({label_name: labels})
    data[label_name] = data[label_name].astype("category")
    x_vars = [f"view 1 projection {f + 1}" for f in range(scores[0].shape[1])]
    y_vars = [f"view 2 projection {f + 1}" for f in range(scores[1].shape[1])]
    data[x_vars] = scores[0]
    data[y_vars] = scores[1]
    cca_pp = sns.pairplot(data, hue=label_name, x_vars=x_vars, y_vars=y_vars)
    cca_pp.fig.suptitle(title)
    return cca_pp


def scatterplot_label(
    scores: np.ndarray, labels=None, label_name=None, title="", ax=None
):
    """
    Makes a scatter plot showing projections coloured by categorical label
    """
    if label_name is None:
        label_name = "label"
    data = pd.DataFrame({label_name: labels})
    data[label_name] = data[label_name].astype("category")
    data["x"] = scores[:, 0]
    data["y"] = scores[:, 1]
    cca_tp = sns.scatterplot(data=data, x="x", y="y", hue=label_name, ax=ax)
    cca_tp.set(title=title)
    return cca_tp


def tsne_label(
    scores: np.ndarray,
    labels=None,
    label_name=None,
    title="",
    verbose=1,
    perplexity=40,
    n_iter=300,
    ax=None,
):
    """
    Makes a tsne plot of the projections from one view with optional labels

    :param scores: projections of data obtained by model.transform(*data)
    :param labels: array of labels
    :param label_name: name of label for legend
    :param title: Figure title
    """
    tsne_scores = TSNE(
        n_components=2, verbose=verbose, perplexity=perplexity, n_iter=n_iter
    ).fit_transform(scores)
    return scatterplot_label(tsne_scores, labels, label_name, title, ax)
