import os

from CCA_methods.generate_data import *
from CCA_methods.linear import *
from CCA_methods.plot_utils import *

plt.style.use('ggplot')
plt.rcParams['lines.markersize'] = 2
outdim_size = 1


def witten_comp(homedir, n, p, q, r, method='witten'):
    os.chdir(homedir + '/scca_comparison_3d/' + method)
    if method == 'witten':
        X, Y, Z, true_x, true_y, true_z = generate_witten_3d(n * 2, outdim_size, p, q, r, sigma=0.7,
                                                             sparse_variables_1=10,
                                                             sparse_variables_2=10, sparse_variables_3=10)
    elif method == 'identity':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='identity', sigma=0.01)
    elif method == 'correlated':
        X, Y, true_x, true_y = generate_mai(n * 2, outdim_size, p, q, sparse_variables_1=10,
                                            sparse_variables_2=10, structure='correlated', sigma=0.1)

    true_x /= true_x.max()
    true_y /= true_y.max()
    true_z /= true_z.max()
    plt.rcParams.update({'font.size': 10})
    plt.figure()
    plt.imshow(X.T @ X)
    plt.colorbar()
    plt.title('Covariance Matrix of X')
    plt.savefig('Covariance_X')

    plt.figure()
    plt.imshow(Y.T @ Y)
    plt.colorbar()
    plt.title('Covariance Matrix of Y')
    plt.savefig('Covariance_Y')

    x_test = X[:n, :]
    y_test = Y[:n, :]
    z_test = Z[:n, :]
    x_train = X[n:, :]
    y_train = Y[n:, :]
    z_train = Z[n:, :]

    x_test -= x_train.mean(axis=0)
    y_test -= y_train.mean(axis=0)
    z_test -= z_train.mean(axis=0)
    x_train -= x_train.mean(axis=0)
    y_train -= y_train.mean(axis=0)
    z_train -= z_train.mean(axis=0)

    max_iter = 10

    # params = {'c_1': 4, 'c_2': 4}
    c1 = [1, 3, 7, 9]
    c2 = [1, 3, 7, 9]
    c3 = [1, 3, 7, 9]

    param_candidates = {'c': list(itertools.product(c1, c2, c3))}
    # param_candidates = {'c_1': [1, 3, 7], 'c_2': [1, 3, 7]}
    Witten = Wrapper(outdim_size=outdim_size, method='pmd', max_iter=max_iter).cv_fit(x_train, y_train, z_train,
                                                                                      param_candidates=param_candidates,
                                                                                      verbose=True, folds=10)

    c1 = [1e-3, 3e-3, 5e-3]
    c2 = [1e-3, 3e-3, 5e-3]
    c3 = [1e-3, 3e-3, 5e-3]

    param_candidates = {'c': list(itertools.product(c1, c2, c3))}
    SCCA = Wrapper(outdim_size=outdim_size, method='scca', max_iter=max_iter).cv_fit(x_train, y_train, z_train,
                                                                                     param_candidates=param_candidates,
                                                                                     verbose=True, folds=10)

    c1 = [0.1, 0.3, 0.5]
    c2 = [0.1, 0.3, 0.5]
    c3 = [0.1, 0.3, 0.5]

    param_candidates = {'c': list(itertools.product(c1, c2, c3))}
    SCCA_constrained = Wrapper(outdim_size=outdim_size, method='constrained_scca', max_iter=max_iter).cv_fit(x_train,
                                                                                                             y_train,
                                                                                                             z_train,
                                                                                                             param_candidates=param_candidates,
                                                                                                             verbose=True, folds=10)

    models = [Witten, SCCA, SCCA_constrained]
    corrs = out_of_sample_corr(models, x_test, y_test, z_test, method)
    witten_plot(models, true_x, true_y, true_z, method)
    return corrs


def witten_plot(models, true_x, true_y, true_z, method):
    plt.rcParams.update({'font.size': 5})
    M = len(models)
    x = np.arange(true_x.shape[0])
    y = np.arange(true_y.shape[0])
    z = np.arange(true_z.shape[0])
    fig, axs = plt.subplots(M + 1, true_x.shape[1] * 3, sharey='row')
    for i in range(true_x.shape[1]):
        non_z_x = (np.abs(true_x) > 1e-9)
        non_z_y = (np.abs(true_y) > 1e-9)
        non_z_z = (np.abs(true_z) > 1e-9)
        axs[0, i].scatter(x[non_z_x[:, i]], true_x[non_z_x[:, i], i])
        axs[0, i].set_title('True weights X_1 dimension ' + str(i))
        axs[0, i + true_x.shape[1]].scatter(y[non_z_y[:, i]], true_y[non_z_y[:, i], i])
        axs[0, i + true_x.shape[1]].set_title('True weights X_2 dimension ' + str(i))
        axs[0, i + 2 * true_x.shape[1]].scatter(z[non_z_z[:, i]], true_z[non_z_z[:, i], i])
        axs[0, i + 2 * true_x.shape[1]].set_title('True weights X_3 dimension ' + str(i))

        axs[0, i].scatter(x[~non_z_x[:, i]], true_x[~non_z_x[:, i], i])
        axs[0, i + true_x.shape[1]].scatter(y[~non_z_y[:, i]], true_y[~non_z_y[:, i], i])
        axs[0, i + 2*true_x.shape[1]].scatter(z[~non_z_z[:, i]], true_z[~non_z_z[:, i], i])
        axs[0, i + true_x.shape[1]].set_ylim(-1.1, 1.1)

    good_W = np.zeros((M, true_x.shape[1], true_x.shape[0]), dtype=bool)
    good_C = np.zeros((M, true_y.shape[1], true_y.shape[0]), dtype=bool)
    good_Q = np.zeros((M, true_z.shape[1], true_z.shape[0]), dtype=bool)
    for m, model in enumerate(models):
        good_W[m, :, :] = (np.abs(true_x) > 1e-9).T
        good_C[m, :, :] = (np.abs(true_y) > 1e-9).T
        good_Q[m, :, :] = (np.abs(true_z) > 1e-9).T
        axs[m + 1, 0].set_ylim(min(model.weights_list[0][:, :].min(), model.weights_list[1][:, :].min(), model.weights_list[2][:, :].min()),
                               max(model.weights_list[0][:, :].max(), model.weights_list[1][:, :].max(), model.weights_list[2][:, :].max()))
        for i in range(true_x.shape[1]):
            # Plot matching
            axs[m + 1, i].scatter(x[good_W[m, 0, :]], model.weights_list[0][good_W[m, 0, :],0] / model.weights_list[0].max())
            axs[m + 1, i].set_title(model.method)
            axs[m + 1, i + true_x.shape[1]].scatter(y[good_C[m, 0, :]], model.weights_list[1][good_C[m, 0, :],0] / model.weights_list[1].max(),
                                                    label='True zero')
            axs[m + 1, i + true_x.shape[1]].set_title(model.method)
            axs[m + 1, i + 2*true_x.shape[1]].scatter(z[good_Q[m, 0, :]], model.weights_list[2][good_C[m, 0, :],0] / model.weights_list[2].max(),
                                                    label='True zero')
            axs[m + 1, i + 2*true_x.shape[1]].set_title(model.method)
            # Plot not-matching
            axs[m + 1, i].scatter(x[~good_W[m, 0, :]], model.weights_list[0][~good_W[m, 0, :],0] / model.weights_list[0].max())
            axs[m + 1, i + true_x.shape[1]].scatter(y[~good_C[m, 0, :]], model.weights_list[1][~good_C[m, 0, :],0] / model.weights_list[1].max(),
                                                    label='True non zero')
            axs[m + 1, i + 2*true_x.shape[1]].scatter(z[~good_Q[m, 0, :]], model.weights_list[2][~good_Q[m, 0, :],0] / model.weights_list[2].max(),
                                                    label='True non zero')
            axs[m + 1, i + true_x.shape[1]].set_ylim(-1.2, 1.2)
    axs[-1, -1].legend(loc='lower right')
    plt.tight_layout()
    plt.rcParams.update({'font.size': 10})
    fig.suptitle('True weights vs model weights')
    fig.subplots_adjust(top=0.88)
    plt.savefig(method + '_sparse_components')


def out_of_sample_corr(models, x_test, y_test, z_test, method):
    labels = [model.method for model in models]
    train_corrs = np.zeros((len(models), 1))
    corrs = np.zeros((len(models), 1))
    for m, model in enumerate(models):
        train_corrs[m] = np.squeeze(model.train_correlations)[np.triu_indices(3,1)].sum()
        corrs[m] = model.predict_corr(x_test, y_test, z_test)[np.triu_indices(3,1)].sum()

    # set width of bar
    barWidth = 0.7
    r = 2 * np.arange(len(labels))
    r1 = [x - barWidth / 2 for x in r]
    r2 = [x + barWidth / 2 for x in r]

    # Make the plot
    fig, ax = plt.subplots()
    ax.bar(r1, train_corrs[:, 0], width=barWidth, edgecolor='white', label='Train')
    ax.bar(r2, corrs[:, 0], width=barWidth, edgecolor='white', label='Test')

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


n = 100
p = 100
q = 100
r = 100

homedir = home_dir = os.getcwd()

# witten_comp(homedir, n, p, q, method='correlated')

# witten_comp(homedir, n, p, q, method='identity')

witten_comp(homedir, n, p, q, r, method='witten')

plt.show()
