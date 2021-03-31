import matplotlib

matplotlib.use('agg')
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

"""
A bunch of methods I have added to help me do plotting when needed

cv_plot() used to automatically generate basic hyperparameter plots for linear wrapper cv_fit() method

plot_results() used to generate comparison plots for HCP data


"""


def cv_plot(scores, param_dict, reg):
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
        print('plot not implemented for more than 4 hyperparameters')
        return
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

    """
    for
    sub_df=

        lineObjects = plt.plot(np.array(sorted(list(x1_vals))),
                               np.squeeze(scores.reshape((len(x1_vals), len(x2_vals), -1)).mean(axis=-1)))
        plt.legend(lineObjects, sorted(list(x2_vals)), title=x2_name)
    """
    fig.suptitle('Hyperparameter plot ' + reg)
    plt.savefig('Hyperparameter_plot_' + reg)


def split_columns(df):
    cols = []
    # check first row to see if each column contains a list or tuple
    for (columnName, columnData) in df.iteritems():
        if isinstance(columnData[0], tuple) or isinstance(columnData[0], list):
            cols.append(columnName)
    for col in cols:
        df = df.join(pd.DataFrame(df[col].tolist()).rename(columns=lambda x: col + '_' + str(x + 1))).drop(col, 1)
    return df


def plot_results(data, labels):
    # data is c*3*k where c is the different models and k is the number of latents and 3 is train,test,val

    # Compare sum of first k dimensions
    corr_sum = np.sum(data, axis=2)

    # set width of bar
    barWidth = 0.7
    r = 2 * np.arange(len(labels))
    r1 = [x - barWidth / 2 for x in r]
    r2 = [x + barWidth / 2 for x in r]

    # Make the plot
    fig, ax = plt.subplots()
    ax.bar(r1, corr_sum[:, 0], width=barWidth, edgecolor='white', label='Train')
    ax.bar(r2, corr_sum[:, 1], width=barWidth, edgecolor='white', label='Test')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('model', fontweight='bold')
    ax.set_ylabel('Sum of first n correlations', fontweight='bold')
    # plt.xticks([r + barWidth for r in range(len(labels))], labels)
    ax.set_xticks(r)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=90)

    # Create legend & Show graphic
    ax.legend()
    fig.tight_layout()
    fig.savefig('compare_train_test')

    # Train dimensions
    plt.figure()
    x = np.arange(1, data.shape[2] + 1)
    for i, m in enumerate(labels):
        if any(m in _ for _ in ['KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA']):
            plt.plot(x, data[i, 0, :], linestyle='dashed')
        else:
            plt.plot(x, data[i, 0, :])
    plt.title('train canonical correlations')
    plt.legend(labels)
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('train_dims')

    # Test dimensions
    plt.figure()
    for i, m in enumerate(labels):
        if any(m in _ for _ in ['KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA']):
            plt.plot(x, data[i, 1, :], linestyle='dashed')
        else:
            plt.plot(x, data[i, 1, :])
    plt.title('test canonical correlations')
    plt.legend(labels)
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('test_dims')


def plot_training_loss(train, val):
    plt.figure()
    plt.plot(train, label='Train')
    plt.plot(val, label='Val')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative correlation sum)')
    plt.legend()
    plt.savefig('training_loss')
