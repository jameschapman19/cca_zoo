import matplotlib.pyplot as plt
import numpy as np

import CCA_methods


def dist_spar_plot(track_distance, track_active_weights_w, track_active_weights_c, params, title, dist='correlation',
                   max_norm=100, max_obj=1):
    fig, axs = plt.subplots(1, track_active_weights_w.shape[0] - 1, sharey=True)
    fig.suptitle(title)
    axs[0].set_ylabel(dist)
    axs_2 = [a.twinx() for a in axs]
    axs_2[-1].set_ylabel('|w|_1')
    for p in range(track_active_weights_w.shape[0] - 1):
        axs[p].plot(track_distance[p, :].T, color='k', label=dist)
        axs[p].set_xlabel('Iterations')
        axs_2[p].plot(track_active_weights_w[p, :].T, label='view 1 weights', linestyle=':')
        axs_2[p].plot(track_active_weights_c[p, :].T, label='view 2 weights', linestyle=':')
        if dist == 'correlation':
            axs[p].set_ylim(bottom=0, top=1.1)
            axs[p].set_xlim(left=0, right=track_distance.shape[1])
        else:
            axs[p].set_ylim(bottom=0, top=max_obj * 1.1)
        axs_2[p].set_ylim(bottom=0, top=max_norm * 1.1)
        axs_2[p].set_xlim(left=0, right=track_distance.shape[1])
        if p == track_active_weights_w.shape[0] - 1:
            axs[p].set_title('Normal ALS')
        else:
            axs[p].set_title('c={:.2E}'.format(params[p]))
        handles, labels = axs_2[p].get_legend_handles_labels()
        handles2, labels2 = axs[p].get_legend_handles_labels()
        handles = handles + handles2
        labels = labels + labels2
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.2)
    plt.savefig("sparse_convergence_init/" + title + "_" + dist)


def convergence_test(X, Y, max_iters=1, parameter_grid=None, method='l2', initialization='random', generalized=True):
    if parameter_grid is None:
        parameter_grid = [[0], [0]]

    length = parameter_grid.shape[0] + 1

    # Bunch of objects to store outputs
    track_correlations = np.empty((length, max_iters))
    track_norms_w = np.zeros((length, max_iters))
    track_norms_c = np.zeros((length, max_iters))
    track_obj = np.zeros((length, max_iters))
    w = np.zeros((length, X.shape[1]))
    c = np.zeros((length, Y.shape[1]))
    # For each of the sets of parameters
    for l in range(length - 1):
        params = {'c': [parameter_grid[l, 0], parameter_grid[l, 1]], 'ratio': [1, 1]}
        # This uses my ALS_inner_loop class
        ALS = CCA_methods.linear.ALS_inner_loop(X, Y, method=method, params=params, max_iter=max_iters,
                                                initialization=initialization, tol=1e-4, generalized=generalized)
        # Get the statistics
        # track_correlations[l, :] = ALS.track_correlation
        # track_norms_w[l, :] = ALS.track_norms_w
        # track_norms_c[l, :] = ALS.track_norms_c
        # Useful for the John one
        track_obj[l, :len(ALS.track_lyuponov)] = ALS.track_lyuponov
        w[l, :] = ALS.weights[0]
        c[l, :] = ALS.weights[1]

    # Run an ALS CCA for comparison
    ALS = CCA_methods.linear.ALS_inner_loop(X, Y, method='l2', params=None, max_iter=max_iters)
    # track_correlations[-1, :] = ALS.track_correlation
    # track_norms_w[-1, :] = ALS.track_norms_w
    # track_norms_c[-1, :] = ALS.track_norms_c
    # track_obj /= np.nanmax(track_obj)
    return track_correlations, track_norms_w, track_norms_c, track_obj


max_iters = 50
X, Y, _, _ = CCA_methods.generate_data.generate_mai(200, 1, 100, 100, sparse_variables_1=20,
                                                    sparse_variables_2=20, structure='toeplitz', sigma=0.95,
                                                    random=True)

# X = np.random.rand(200, 100)
# Y = np.random.rand(200, 100)

X -= X.mean(axis=0)
Y -= Y.mean(axis=0)

X /= X.std(axis=0)
Y /= Y.std(axis=0)

number_of_parameters_to_try = 5

### SCCA vs. SGCCA
parameter_grid = np.ones((number_of_parameters_to_try, 2)) * 0
parameter_grid[:, 0] = np.linspace(1e-5, 1e-4, number_of_parameters_to_try)
parameter_grid[:, 1] = np.linspace(1e-5, 1e-4, number_of_parameters_to_try)

elastic_gen = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='elastic',
                               generalized=True)

elastic_jc_gen = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='elastic_jc',
                                  generalized=True)

elastic = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='elastic',
                           generalized=False)

elastic_jc = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='elastic_jc',
                              generalized=False)

max_norm = max(np.nanmax(elastic[1]), np.nanmax(elastic_jc[1]),
               np.nanmax(elastic_gen[1]), np.nanmax(elastic_jc_gen[1]),
               np.nanmax(elastic[2]), np.nanmax(elastic_jc[2]),
               np.nanmax(elastic_gen[2]), np.nanmax(elastic_jc_gen[2]))

max_obj = max(np.nanmax(elastic[3]), np.nanmax(elastic_jc[3]),
              np.nanmax(elastic_gen[3]), np.nanmax(elastic_jc_gen[3]))

dist_spar_plot(elastic[3], elastic[1], elastic[2], parameter_grid[:, 0], 'Elastic', dist='Objective Function',
               max_norm=max_norm, max_obj=max_obj)

dist_spar_plot(elastic_jc[3], elastic_jc[1], elastic_jc[2], parameter_grid[:, 0], 'Elastic James',
               dist='Objective Function',
               max_norm=max_norm, max_obj=max_obj)

dist_spar_plot(elastic_gen[3], elastic_gen[1], elastic_gen[2], parameter_grid[:, 0], 'Elastic Generalized',
               dist='Objective Function',
               max_norm=max_norm, max_obj=max_obj)

dist_spar_plot(elastic_jc_gen[3], elastic_jc_gen[1], elastic_jc_gen[2], parameter_grid[:, 0],
               'Elastic Generalized James', dist='Objective Function',
               max_norm=max_norm, max_obj=max_obj)

print('here')
# dist_spar_plot(scca[3], scca[1], scca[2], parameter_grid[:, 0], 'Elastic Waaijenborg', dist='Objective Function',
#               max_norm=max_norm, max_obj=max_obj)

# dist_spar_plot(scca_random[3], scca_random[1], scca_random[2], parameter_grid[:, 0], 'Elastic JC',
#               dist='Objective Function',
#               max_norm=max_norm, max_obj=max_obj)
