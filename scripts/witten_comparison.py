import itertools
import os

from CCA_methods.generate_data import *
from CCA_methods.linear import *
from CCA_methods.plot_utils import *

plt.style.use('ggplot')
plt.rcParams['lines.markersize'] = 2
outdim_size = 1


def witten_comp(homedir, n, p, q, method='witten'):
    os.chdir(homedir + '/scca_comparison_tree/' + method)
    if method == 'witten':
        X, Y, true_x, true_y = generate_witten(n * 2, outdim_size, p, q, sigma=0.3, tau=0.3, sparse_variables_1=10,
                                               sparse_variables_2=10)
    elif method == 'identity':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='identity', sigma=0.1)
    elif method == 'correlated':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='correlated', sigma=0.05)
    elif method == 'toeplitz':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='toeplitz', sigma=0.8)
    elif method == 'toeplitz_high':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='toeplitz', sigma=0.95)

    true_x /= true_x.max()
    true_y /= true_y.max()
    plt.rcParams.update({'font.size': 10})
    plt.figure()
    plt.imshow(X.T @ Y)
    plt.colorbar()
    plt.title('Cross-covariance Matrix: ' + method)
    plt.savefig('Covariance_X' + method)

    plt.figure()
    plt.imshow(Y.T @ Y)
    plt.colorbar()
    plt.title('Covariance Matrix of Y')
    plt.savefig('Covariance_Y')

    x_test = X[:n, :]
    y_test = Y[:n, :]
    x_train = X[n:, :]
    y_train = Y[n:, :]

    x_test -= x_train.mean(axis=0)
    y_test -= y_train.mean(axis=0)
    x_train -= x_train.mean(axis=0)
    y_train -= y_train.mean(axis=0)

    max_iter = 20
    cv_folds = 5

    c1 = [4, 5, 6]
    c2 = [4, 5, 6]
    param_candidates = {'c': list(itertools.product(c1, c2))}

    treeCCA = Wrapper(outdim_size=outdim_size, method='tree_jc', max_iter=max_iter).cv_fit(x_train, y_train,
                                                                                           param_candidates=param_candidates,
                                                                                           verbose=True, folds=cv_folds)

    c1 = [0.1, 0.2, 0.3, 0.4]
    c2 = [0.1, 0.2, 0.3, 0.4]
    param_candidates = {'c': list(itertools.product(c1, c2))}

    SCCA_constrained = Wrapper(outdim_size=outdim_size, method='constrained_scca', max_iter=max_iter,
                               generalized=False).cv_fit(x_train,
                                                         y_train,
                                                         param_candidates=param_candidates,
                                                         verbose=True,
                                                         folds=cv_folds)

    SGCCA_constrained = Wrapper(outdim_size=outdim_size, method='constrained_scca', max_iter=max_iter).cv_fit(x_train,
                                                                                                              y_train,
                                                                                                              param_candidates=param_candidates,
                                                                                                              verbose=True,
                                                                                                              folds=cv_folds)

    c1 = [1, 3, 7]
    c2 = [1, 3, 7]
    param_candidates = {'c': list(itertools.product(c1, c2))}
    Witten = Wrapper(outdim_size=outdim_size, method='pmd', max_iter=max_iter).cv_fit(x_train, y_train,
                                                                                      param_candidates=param_candidates,
                                                                                      verbose=True, folds=cv_folds)

    c1 = [3e-3, 4e-3, 5e-3]
    c2 = [3e-3, 4e-3, 5e-3]
    param_candidates = {'c': list(itertools.product(c1, c2))}

    SCCA = Wrapper(outdim_size=outdim_size, method='scca', max_iter=max_iter, generalized=False).cv_fit(x_train,
                                                                                                        y_train,
                                                                                                        param_candidates=param_candidates,
                                                                                                        verbose=True,
                                                                                                        folds=cv_folds)

    SGCCA = Wrapper(outdim_size=outdim_size, method='scca', max_iter=max_iter).cv_fit(x_train, y_train,
                                                                                      param_candidates=param_candidates,
                                                                                      verbose=True, folds=cv_folds)

    CCA = Wrapper(outdim_size=outdim_size, method='pls', max_iter=max_iter).fit(x_train, y_train)

    models = [treeCCA, Witten, SCCA, SGCCA, SCCA_constrained, SGCCA_constrained, CCA]
    witten_plot(models, true_x, true_y, method)

    true_train = np.corrcoef((x_train @ true_x).T, (y_train @ true_y).T)[0, 1]
    true_test = np.corrcoef((x_test @ true_x).T, (y_test @ true_y).T)[0, 1]
    corrs = out_of_sample_corr(models, x_test, y_test, method, true_train=true_train, true_test=true_test)
    return corrs


def witten_plot(models, true_x, true_y, method):
    plt.rcParams.update({'font.size': 5})
    M = len(models)
    x = np.arange(true_x.shape[0])
    y = np.arange(true_y.shape[0])
    fig, axs = plt.subplots(M + 1, true_x.shape[1] * 2, sharey='row')
    for i in range(true_x.shape[1]):
        non_z_x = (np.abs(true_x) > 1e-9)
        non_z_y = (np.abs(true_y) > 1e-9)
        axs[0, i].scatter(x[non_z_x[:, i]], true_x[non_z_x[:, i], i])
        axs[0, i].set_title('True weights X_1 dimension ' + str(i))
        axs[0, i + true_x.shape[1]].scatter(y[non_z_y[:, i]], true_y[non_z_y[:, i], i])
        axs[0, i + true_x.shape[1]].set_title('True weights X_2 dimension ' + str(i))
        axs[0, i].scatter(x[~non_z_x[:, i]], true_x[~non_z_x[:, i], i])
        axs[0, i + true_x.shape[1]].scatter(y[~non_z_y[:, i]], true_y[~non_z_y[:, i], i])
        axs[0, i + true_x.shape[1]].set_ylim(-1.1, 1.1)
    good_W = np.zeros((M, true_x.shape[1], true_x.shape[0]), dtype=bool)
    good_C = np.zeros((M, true_y.shape[1], true_y.shape[0]), dtype=bool)
    for m, model in enumerate(models):
        good_W[m, :, :] = (np.abs(true_x) > 1e-9).T
        good_C[m, :, :] = (np.abs(true_y) > 1e-9).T
        axs[m + 1, 0].set_ylim(min(model.weights_list[0][:, :].min(), model.weights_list[1][:, :].min()),
                               max(model.weights_list[0][:, :].max(), model.weights_list[1][:, :].max()))
        for i in range(true_x.shape[1]):
            # Plot matching
            axs[m + 1, i].scatter(x[good_W[m, 0, :]],
                                  model.weights_list[0][good_W[m, 0, :]] / model.weights_list[0].max())
            axs[m + 1, i].set_title(model.method)
            # axs[m + 1, 1].scatter(x[good_W[m, 1, :]], model.W[0, good_W[m, 1, :]])
            # axs[m + 1, 1].set_title(model.method)
            axs[m + 1, i + true_x.shape[1]].scatter(y[good_C[m, 0, :]],
                                                    model.weights_list[1][good_C[m, 0, :]] / model.weights_list[
                                                        1].max(),
                                                    label='True zero')
            axs[m + 1, i + true_x.shape[1]].set_title(model.method)
            # axs[m + 1, 3].scatter(x[good_C[m, 1, :]], model.C[0, good_C[m, 1, :]])
            # axs[m + 1, 3].set_title(model.method)
            # Plot not-matching
            axs[m + 1, i].scatter(x[~good_W[m, 0, :]],
                                  model.weights_list[0][~good_W[m, 0, :]] / model.weights_list[0].max())
            # axs[m + 1, 1].scatter(x[~good_W[m, 1, :]], model.W[0, ~good_W[m, 1, :]])
            axs[m + 1, i + true_x.shape[1]].scatter(y[~good_C[m, 0, :]],
                                                    model.weights_list[1][~good_C[m, 0, :]] / model.weights_list[
                                                        1].max(),
                                                    label='True non zero')
            # axs[m + 1, 3].scatter(x[~good_C[m, 1, :]], model.C[0, ~good_C[m, 1, :]])
            axs[m + 1, i + true_x.shape[1]].set_ylim(-1.2, 1.2)
    axs[-1, -1].legend(loc='lower right')
    plt.tight_layout()
    plt.rcParams.update({'font.size': 10})
    fig.suptitle('True weights vs model weights')
    fig.subplots_adjust(top=0.88)
    plt.savefig(method + '_sparse_components')


def out_of_sample_corr(models, x_test, y_test, method, true_train, true_test):
    labels = [model.method for model in models]
    train_corrs = np.zeros((len(models), 1))
    corrs = np.zeros((len(models), 1))
    for m, model in enumerate(models):
        train_corrs[m] = model.train_correlations[0, 1, 0]
        corrs[m] = model.predict_corr(x_test, y_test)[0, 1, 0]

    train_corrs = np.append(train_corrs, true_train)
    corrs = np.append(corrs, true_test)
    labels.append('True')
    # set width of bar
    barWidth = 0.7
    r = 2 * np.arange(len(labels))
    r1 = [x - barWidth / 2 for x in r]
    r2 = [x + barWidth / 2 for x in r]

    # Make the plot
    fig, ax = plt.subplots()
    ax.bar(r1, train_corrs, width=barWidth, edgecolor='white', label='Train')
    ax.bar(r2, corrs, width=barWidth, edgecolor='white', label='Test')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('model', fontweight='bold')
    # plt.xticks([r + barWidth for r in range(len(labels))], labels)
    ax.set_xticks(r)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=90)

    # Create legend & Show graphic
    ax.legend()
    ax.set_title('Train and test Correlations by Model')
    ax.set_ylabel('Correlation')
    fig.tight_layout()
    plt.savefig(method + '_corrs')

    return corrs


n = 200
p = 200
q = 200

homedir = home_dir = os.getcwd()

witten_comp(homedir, n, p, q, method='toeplitz')

witten_comp(homedir, n, p, q, method='toeplitz_high')

witten_comp(homedir, n, p, q, method='witten')

witten_comp(homedir, n, p, q, method='identity')

plt.show()
