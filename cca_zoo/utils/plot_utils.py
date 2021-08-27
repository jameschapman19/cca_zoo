import itertools
from typing import Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm


def cv_plot(scores, param_dict, model_name):
    """
    Plot a hyperparameter surface plot (or multiple surface plots if more than 2 hyperparameters

    :param scores:
    :param param_dict:
    :param model_name:
    """
    scores = pd.Series(scores)
    hyper_df = pd.DataFrame(param_dict)
    hyper_df = split_columns(hyper_df)
    hyper_df = hyper_df[[i for i in hyper_df if len(set(hyper_df[i])) > 1]]
    # Check dimensions
    dimensions = len(hyper_df.columns)
    n_uniques = hyper_df.nunique()
    sub_dfs = []
    sub_scores = []
    if dimensions > 4:
        raise ValueError('plot not implemented for more than 4 hyperparameters')
    elif dimensions == 4:
        fig, axs = plt.subplots(n_uniques[-2], n_uniques[-1], subplot_kw={'projection': '3d'})
        unique_x = hyper_df[hyper_df.columns[-2]].unique()
        unique_y = hyper_df[hyper_df.columns[-1]].unique()
        param_pairs = list(itertools.product(unique_x, unique_y))
        for pair in param_pairs:
            mask = (hyper_df[hyper_df.columns[-2]] == pair[0]) & (hyper_df[hyper_df.columns[-1]] == pair[1])
            sub_dfs.append(hyper_df.loc[mask].iloc[:, :-2])
            sub_scores.append(scores[mask])
    elif dimensions == 3:
        fig, axs = plt.subplots(1, n_uniques[-1], subplot_kw={'projection': '3d'})
        unique_x = hyper_df[hyper_df.columns[-1]].unique()
        for x in unique_x:
            mask = (hyper_df[hyper_df.columns[-1]] == x)
            sub_dfs.append(hyper_df.loc[mask].iloc[:, :-1])
            sub_scores.append(scores[mask])
    else:
        sub_dfs.append(hyper_df)
        sub_scores.append(scores)
        if dimensions == 2:
            fig, axs = plt.subplots(1, subplot_kw={'projection': '3d'})
        else:
            fig, axs = plt.subplots(1)
        axs = np.array([axs])
    axs = axs.flatten()
    for i, (ax, sub_df, sub_score) in enumerate(zip(axs, sub_dfs, sub_scores)):
        if len(sub_df.shape) > 1:
            ax.plot_trisurf(np.log(sub_df.iloc[:, 0]), np.log(sub_df.iloc[:, 1]), sub_score, cmap=cm.coolwarm,
                            linewidth=0.2)
            ax.set_xlabel('log ' + sub_df.columns[0])
            ax.set_ylabel('log ' + sub_df.columns[1])
            ax.set_zlabel('Sum of first n correlations')

    fig.suptitle('Hyperparameter plot ' + model_name)
    return fig


def split_columns(df):
    cols = []
    # check first row to see if each column contains a list or tuple
    for (columnName, columnData) in df.iteritems():
        if isinstance(columnData[0], tuple) or isinstance(columnData[0], list):
            cols.append(columnName)
    for col in cols:
        df = df.join(pd.DataFrame(df[col].tolist()).rename(columns=lambda x: col + '_' + str(x + 1))).drop(col, axis=1)
    return df


def plot_latent_train_test(train_scores: Union[Tuple[np.ndarray], List[np.ndarray]],
                           test_scores: Union[Tuple[np.ndarray], List[np.ndarray]], title=''):
    """
    Makes a pair plot showing the projections of each view against each other for each dimensions. Coloured by train and test

    :param train_scores: projections of training data which can be accessed by model.scores
    :param test_scores: projections of test data obtained by model.transform(*test_data)
    :param title: Figure title
    """
    train_data = pd.DataFrame(
        {'phase': np.asarray(['train'] * train_scores[0].shape[0]).astype(str)})
    x_vars = [f'view 1 projection {f}' for f in range(train_scores[0].shape[1])]
    y_vars = [f'view 2 projection {f}' for f in range(train_scores[1].shape[1])]
    train_data[x_vars] = train_scores[0]
    train_data[y_vars] = train_scores[1]
    test_data = pd.DataFrame(
        {'phase': np.asarray(['test'] * test_scores[0].shape[0]).astype(str)})
    test_data[x_vars] = test_scores[0]
    test_data[y_vars] = test_scores[1]
    data = pd.concat([train_data, test_data], axis=0)
    cca_pp = sns.pairplot(data, hue='phase', x_vars=x_vars, y_vars=y_vars, corner=True)
    cca_pp.fig.suptitle(title)
    return cca_pp


def plot_latent_label(scores: Union[Tuple[np.ndarray], List[np.ndarray]], labels=None, label_name=None, title=''):
    """
    Makes a pair plot showing the projections of each view against each other for each dimensions. Coloured by categorical label

    :param scores: projections of data obtained by model.transform(*data)
    :param labels: array of labels
    :param label_name: name of label for legend
    :param title: Figure title
    """
    if label_name is None:
        label_name = 'label'
    data = pd.DataFrame(
        {label_name: labels.astype(str)})
    x_vars = [f'view 1 projection {f}' for f in range(scores[0].shape[1])]
    y_vars = [f'view 2 projection {f}' for f in range(scores[1].shape[1])]
    data[x_vars] = scores[0]
    data[y_vars] = scores[1]
    cca_pp = sns.pairplot(data, hue=label_name, x_vars=x_vars, y_vars=y_vars, corner=True)
    cca_pp.fig.suptitle(title)
    return cca_pp
