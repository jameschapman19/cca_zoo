import os

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
import scipy.io as sio
import scipy.stats as ss
from sklearn.decomposition import PCA


def replace_nan_column_mean(A):
    col_mean = np.nanmean(A, axis=0)
    inds = np.where(np.isnan(A))
    A[inds] = np.take(col_mean, inds[1])
    return A

def reduce_dimensions(train_x, train_y, test_x=None, test_y=None, components_x=100, components_y=100):
    pca = PCA(n_components=components_x)
    train_x = pca.fit_transform(train_x)
    if not test_x is None:
        test_x = pca.transform(test_x)

    pca = PCA(n_components=components_y)
    train_y = pca.fit_transform(train_y)
    if not test_y is None:
        test_y = pca.transform(test_y)
    return train_x, train_y, test_x, test_y

def find_categoricals(data, categories, data_dict):
    clinical_entries = data_dict[data_dict.category.isin(categories).values]
    a = list(clinical_entries.columnHeader.values)
    cat_columns = data[[c for c in data.columns if c in a]]
    return cat_columns

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A James/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


# Rank based INT

"""The MIT License (MIT)

Copyright (c) 2016 Edward Mountjoy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def rank_INT(array, c=3.0 / 8, stochastic=True):
    # Set seed
    np.random.seed(123)

    nan_bool = np.isnan(array)

    # Drop NaNs
    rank = array[~nan_bool]

    # Get ranks
    if stochastic == True:
        rank = ss.rankdata(rank, method="ordinal")
    else:
        rank = ss.rankdata(rank, method="average")

    # Convert rank to normal distribution
    transformed = rank_to_normal(rank, c=c, n=len(rank))
    array[~nan_bool] = transformed
    return array


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return ss.norm.ppf(x)


def load_hcp_data(subjects='1000', smith=True):
    if os.getcwd() == '/Users/jameschapman/PycharmProjects/MultiView':
        if subjects == '500':
            ids = sio.loadmat('../Data/id_500.mat')['id']
            brain_ids = ids[:, 2] - 1
            behaviour_ids = ids[:, 1]
            conf_ids = ids[:, 0] - 1
        else:
            ids = sio.loadmat('../Data/id_1000.mat')['id']
            brain_ids = ids[:, 2] - 1
            behaviour_ids = ids[:, 1]
            conf_ids = ids[:, 0] - 1
        unres_df = pd.read_excel(
            'Data/Unrestricted_behavioural_data.xlsx')
        print("unres_df:", unres_df.shape)
        res_df = pd.read_excel(
            'Data/Restricted_behavioural_data.xlsx')
        print("res_df:", res_df.shape)
        smith_cats = pd.read_csv('../Data/smith_categories.csv')
        connectivity_matrix = np.random.rand(1003, 200 * 200)
        print("connect:", connectivity_matrix.shape)
        varsQconf = np.loadtxt('Data/varsQconf_1000.txt')
        data_dict = pd.read_csv('../Data/datadict.csv', encoding="ISO-8859-1")
    else:
        if subjects == '500':
            ids = sio.loadmat('/SAN/medic/human-connectome/experiments/hcp_cca_replication/id_500.mat')['id']
            brain_ids = ids[:, 2] - 1
            behaviour_ids = ids[:, 1]
            conf_ids = ids[:, 0] - 1
            varsQconf = np.loadtxt('/SAN/medic/human-connectome/experiments/hcp_cca_replication/varsQconf_500.txt')
            varsrfmri = np.loadtxt('/SAN/medic/human-connectome/experiments/hcp_cca_replication/rfMRI_motion_500.txt')
            varsQconf = np.stack((varsQconf, varsrfmri), -1)
        else:
            ids = sio.loadmat('/SAN/medic/human-connectome/experiments/hcp_cca_replication/id_1000.mat')['id']
            brain_ids = ids[:, 2] - 1
            behaviour_ids = ids[:, 1]
            conf_ids = ids[:, 0] - 1
            varsQconf = np.loadtxt('/SAN/medic/human-connectome/experiments/hcp_cca_replication/varsQconf_1000.txt')
        unres_df = pd.read_excel(
            '/SAN/medic/human-connectome/experiments/hcp_cca_replication/Unrestricted_behavioural_data.xlsx')
        print("unres_df:", unres_df.shape)
        res_df = pd.read_excel(
            '/SAN/medic/human-connectome/experiments/hcp_cca_replication/Restricted_behavioural_data.xlsx')
        print("res_df:", res_df.shape)
        smith_cats = pd.read_csv('/home/jchapman/smith_categories.csv')
        matrix_file_path = '/SAN/medic/human-connectome/RAW/HCP1200Parcellation+Timeseries+Netmats/HCP1200_Parcellation_Timeseries_Netmats_1003s_r177_r227/HCP_PTN1200/netmats_3T_HCP1200_MSMAll_ICAd200_ts2/netmats/3T_HCP1200_MSMAll_d200_ts2/netmats2.txt'
        connectivity_matrix = np.loadtxt(matrix_file_path)
        print("connect:", connectivity_matrix.shape)
        data_dict = pd.read_csv('/home/jchapman/datadict.csv', encoding="ISO-8859-1")


    clinical_cats = [
        'Sensory',
        'Substance Use',
        'Psychiatric and Life Function',
        'Personality',
        'Motor',
        'MEG Subjects',
        'In-Scanner Task Performance',
        'Health and Family History',
        'Emotion',
        'Cognition',
        'Alertness'
    ]

    varsQconf = varsQconf[conf_ids, :]
    connectivity_matrix = connectivity_matrix[brain_ids, :]
    original_brain = connectivity_matrix
    res_df['Subject'] = unres_df['Subject']
    cols_to_use = unres_df.columns.difference(res_df.columns).tolist()
    cols_to_use.append('Subject')
    all_behaviour = pd.merge(res_df, unres_df[cols_to_use], on='Subject')
    all_behaviour = all_behaviour[all_behaviour.Subject.isin(behaviour_ids)]
    original_behaviour = all_behaviour
    original_behaviour = find_categoricals(original_behaviour, clinical_cats, data_dict)
    conf_columns = ['rfMRI_motion', 'Height', 'Weight', 'BPSystolic', 'BPDiastolic', 'HbA1C']
    conf_columns = np.intersect1d(conf_columns, all_behaviour.columns.values)
    cubic_conf_columns = ['FS_IntraCranial_Vol', 'FS_BrainSeg_Vol']
    cubic_conf_columns = np.intersect1d(cubic_conf_columns, all_behaviour.columns.values)

    if smith:
        conf = np.hstack(
            (varsQconf, all_behaviour[conf_columns], all_behaviour[cubic_conf_columns] ** (1 / 3)))
        conf = np.apply_along_axis(rank_INT, 1, conf, c=3.0 / 8, stochastic=True)
        conf[np.isnan(conf)] = 0
        # Add on squared terms and renormalise
        conf = np.hstack((conf, conf[:, varsQconf.shape[1]:] ** 2))
        conf -= conf.mean(axis=0)
        conf /= conf.std(axis=0)
        # Need to do the triu piece again
        dummy_connectivity = np.zeros((200, 200))
        dummy_connectivity[np.triu_indices(200, 1)] = 1
        dummy_connectivity = dummy_connectivity.flatten()
        connectivity_matrix = connectivity_matrix[:, dummy_connectivity == 1]
        # prepare main netmat matrix - we have a range of normalisation possibilities
        NET1 = connectivity_matrix - connectivity_matrix.mean(axis=0)
        NET1 /= NET1.std(axis=0)
        amNET = np.abs(np.mean(connectivity_matrix, axis=0))
        NET3 = connectivity_matrix / amNET
        NET3 -= NET3.mean(axis=0)
        NET3 = NET3[:, amNET > 0.1]
        NET3 /= np.std(NET3)
        grot = np.hstack((NET1, NET3))
        brain = grot - conf @ (np.linalg.pinv(conf) @ grot)
        brain -= brain.mean(axis=0)

        all_behaviour = all_behaviour[smith_cats.Variable.values].astype(np.float64).values
        print("all_behaviour:", all_behaviour.shape)
        behaviour = np.apply_along_axis(rank_INT, 1, all_behaviour, c=3.0 / 8, stochastic=True)

        for i in range(behaviour.shape[1]):
            nan_bool = np.isnan(behaviour[:, i])
            grotconf = conf[~nan_bool, :] - conf[~nan_bool, :].mean()
            rep = behaviour[~nan_bool, i] - grotconf @ (np.linalg.pinv(grotconf) @ behaviour[~nan_bool, i])
            rep -= rep.mean(axis=0)
            rep /= rep.std(axis=0)
            behaviour[~nan_bool, i] = rep
        behaviour_cov = np.zeros((behaviour.shape[0], behaviour.shape[0]))
        for i in range(behaviour.shape[0]):
            for j in range(behaviour.shape[0]):
                grot = behaviour[[i, j], :]
                grot = np.cov(grot[:, (np.sum(np.isnan(grot), axis=0) == 0)])
                behaviour_cov[i, j] = grot[0, 1]

        # project to nearest cov matrix
        behaviour = nearestPD(behaviour_cov)
        brain = replace_nan_column_mean(brain)
        behaviour = replace_nan_column_mean(behaviour)
        return conf, brain, behaviour, original_brain, original_behaviour, connectivity_matrix,data_dict, smith_cats
    else:
        conf = np.hstack((varsQconf, all_behaviour[conf_columns], all_behaviour[cubic_conf_columns]))
        conf = replace_nan_column_mean(conf)
        conf -= conf.mean(axis=0)
        conf /= conf.std(axis=0)
        dummy_connectivity = np.zeros((200, 200))
        dummy_connectivity[np.triu_indices(200, 1)] = 1
        dummy_connectivity = dummy_connectivity.flatten()
        connectivity_matrix = connectivity_matrix[:, dummy_connectivity == 1]
        brain=connectivity_matrix.copy()
        behaviour = all_behaviour[smith_cats.Variable.values].astype(np.float64).values
        behaviour = np.apply_along_axis(rank_INT, 1, behaviour, c=3.0 / 8, stochastic=True)
        behaviour = replace_nan_column_mean(behaviour)
        behaviour -= behaviour.mean(axis=0)
        behaviour /= behaviour.std(axis=0)
        return conf,brain, behaviour, original_brain, original_behaviour, connectivity_matrix,data_dict, smith_cats


def score_feat_corr(U, V, connectivity_matrix, behaviour_measures, data_dict, smith_cats):
    behaviour_measures = behaviour_measures.select_dtypes(include=[np.number])
    behaviour_corr = np.zeros((U.shape[0], behaviour_measures.shape[1]))
    brain_corr = np.zeros((V.shape[0], connectivity_matrix.shape[1]))
    behaviour_columns = behaviour_measures.columns.values
    lookup = dict(zip(data_dict.columnHeader, data_dict.fullDisplayName))
    behaviour_columns_full_names = [lookup[x] for x in behaviour_columns]
    behaviour_measures = replace_nan_column_mean(behaviour_measures.values)
    connectivity_matrix = replace_nan_column_mean(connectivity_matrix)
    for k in range(U.shape[0]):
        behaviour_corr[k, :] = np.corrcoef(V[k], behaviour_measures.T)[1:, 0]
        brain_corr[k, :] = np.corrcoef(U[k], connectivity_matrix.T)[1:, 0]

    mean_edge_connectivity = np.mean(connectivity_matrix, axis=0)
    edge_strength_modulation = np.sign(mean_edge_connectivity) * brain_corr[0]

    ordered_connectivity = np.ones((200, 200))
    ordered_connectivity[np.triu_indices(200, 1)] = mean_edge_connectivity
    ordered_connectivity[np.triu_indices(200, 1)[::-1]] = mean_edge_connectivity

    cca_connectivity = np.ones((200, 200))
    cca_connectivity[np.triu_indices(200, 1)] = edge_strength_modulation
    cca_connectivity[np.triu_indices(200, 1)[::-1]] = edge_strength_modulation

    # Use this to get the ordering from connectivity
    pdist = spc.distance.pdist(ordered_connectivity)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance') - 1

    ordered_connectivity = ordered_connectivity[idx, :][:, idx]
    cca_connectivity = cca_connectivity[idx, :][:, idx]

    behaviour_output = pd.DataFrame(data=behaviour_corr.T, index=behaviour_columns_full_names,
                                    columns=['Correlation_' + str(k) for k in range(behaviour_corr.shape[0])])

    # explained variance
    u = np.expand_dims(U[0, :], 1)
    diff_avg = np.average(u - behaviour_measures, axis=0)
    numerator = np.average((u - behaviour_measures - diff_avg) ** 2, axis=0)

    y_true_avg = np.average(u, axis=0)
    denominator = np.average((u - y_true_avg) ** 2, axis=0)

    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator

    output_scores = np.ones(behaviour_measures.shape[1])

    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator)

    output_scores[~valid_score] = 0

    behaviour_output['Variance explained'] = output_scores

    behaviour_output['In CCA_archive'] = behaviour_output.index.isin(smith_cats.Variable.values)

    behaviour_output = behaviour_output.sort_values('Correlation_0')

    behaviour_output = behaviour_output.dropna()

    return ordered_connectivity, cca_connectivity, linkage, behaviour_output
