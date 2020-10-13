import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def all_experiment_summary(directory='Results/1000_10_runs/'):
    train = []
    test = []
    exp_num = 0
    for experiment in os.listdir(directory):
        if experiment[:10] == 'sparse_run':
            exp_num += 1
            print(directory + experiment + '/results.npy')
            experiment = np.load(directory + experiment + '/results.npy')
            # experiment = np.random.rand(8, 2, 2)
            experiment = np.sum(experiment, axis=2)
            train.extend(experiment[:, 0])
            test.extend(experiment[:, 1])

    train_label = ['train'] * 12 * exp_num
    test_label = ['test'] * 12 * exp_num
    method = ['PLS', 'Ridge - ALS', 'PMD', 'Parkhomenko', 'Sparse - ALS', 'Sparse - Generalized ALS',
              'Elastic - ALS', 'KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA'] * exp_num

    train_df = pd.DataFrame({'train/test': train_label, 'method': method, 'correlation': train})
    test_df = pd.DataFrame({'train/test': test_label, 'method': method, 'correlation': test})
    df = train_df.append(test_df)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.violinplot(x="method", y="correlation", hue="train/test", data=df).set_title(
        "Train and Test Correlations over 10 experiments")
    # palette="Pastel1"
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.tight_layout()
    plt.savefig(directory + 'figures/train_test_distribution')


def all_experiment_by_dimension(directory='Results/1000_10_runs/'):
    dim_1 = []
    dim_2 = []
    #dim_3 = []
    # dim_4 = []
    exp_num = 0
    for experiment in os.listdir(directory):
        if experiment[:10] == 'sparse_run':
            exp_num += 1
            print(directory + experiment + '/results.npy')
            experiment = np.load(directory + experiment + '/results.npy')
            # experiment = np.random.rand(8, 2, 2)
            # experiment = np.sum(experiment, axis=2)
            experiment = experiment[:, 1, :]
            dim_1.extend(experiment[:, 0])
            dim_2.extend(experiment[:, 1])
            #dim_3.extend(experiment[:, 2])
            # dim_4.extend(experiment[:, 3])

    dim_1_label = ['dimension 1'] * 12 * exp_num
    dim_2_label = ['dimension 2'] * 12 * exp_num
    #dim_3_label = ['dimension 3'] * 13 * exp_num
    #dim_4_label = ['dimension 4'] * 13 * exp_num
    method = ['PLS', 'Ridge - ALS', 'PMD', 'Parkhomenko', 'Sparse - ALS', 'Sparse - Generalized ALS',
              'Elastic - ALS', 'KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA'] * exp_num

    dim1_df = pd.DataFrame({'dimension': dim_1_label, 'method': method, 'correlation': dim_1})
    dim2_df = pd.DataFrame({'dimension': dim_2_label, 'method': method, 'correlation': dim_2})
    #dim3_df = pd.DataFrame({'dimension': dim_3_label, 'method': method, 'correlation': dim_3})
    # dim4_df = pd.DataFrame({'dimension': dim_4_label, 'method': method, 'correlation': dim_4})
    df = dim1_df.append(dim2_df)#.append(dim3_df)  # .append(dim4_df)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.violinplot(x="method", y="correlation", hue="dimension", data=df).set_title(
        "Test Correlations over 10 experiments")
    # palette="Pastel1"
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.tight_layout()
    plt.savefig(directory + 'figures/test_dimension_distributions')


def all_w_weights(directory='Results/1000_10_runs/', dim=0):
    w_weights = []
    c_weights = []
    exp_num = 0
    method = ['PLS', 'Ridge - ALS', 'PMD', 'Parkhomenko', 'Sparse - ALS', 'Sparse - Generalized ALS',
              'Elastic - ALS']
    for i in range(len(method)):
        for experiment in os.listdir(directory):
            if experiment[:10] == 'sparse_run':
                exp_num += 1
                print(directory + experiment + '/w_weights.npy')
                data = np.load(directory + experiment + '/w_weights.npy')
                w_weights.extend(data[i, dim, :100])
                data = np.load(directory + experiment + '/c_weights.npy')
                c_weights.extend(data[i, dim, :145])

        c_label = ['c'] * 145 * exp_num
        w_label = ['w'] * 100 * exp_num
        c_feature = list(range(1, 146)) * exp_num
        w_feature = list(range(1, 101)) * exp_num

        w_df = pd.DataFrame({'w/c': w_label, 'feature': w_feature, 'weight': np.abs(w_weights)})
        c_df = pd.DataFrame({'w/c': c_label, 'feature': c_feature, 'weight': np.abs(c_weights)})
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.boxplot(x="feature", y="weight", data=w_df).set_title(
            method[i] + " Brain Weights distribution over experiments (dimension " + str(dim) + ")")
        ax.set_xlabel('')
        ax.set_xticks([])
        plt.tight_layout()
        plt.savefig(directory + 'figures/' + method[i] + '_brain_' + str(dim))
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.boxplot(x="feature", y="weight", data=c_df).set_title(
            method[i] + " Behaviour Weights distribution over experiments (dimension " + str(dim) + ")")
        ax.set_xlabel('')
        ax.set_xticks([])
        plt.tight_layout()
        plt.savefig(directory + 'figures/' + method[i] + '_behaviour_' + str(dim))


directory = 'Results/1000_runs_sparse_final/'

#all_experiment_summary(directory)

#all_experiment_by_dimension(directory)

all_w_weights(directory)

#all_w_weights(directory, dim=1)

#all_w_weights(directory, dim=2)


