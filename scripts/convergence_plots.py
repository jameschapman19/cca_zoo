import matplotlib
import CCA_methods
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')



def dist_spar_plot(track_distance, track_active_weights_w, track_active_weights_c, params, title, dist='correlation',
                   max_norm=100, max_obj=1):
    fig, axs = plt.subplots(1, track_active_weights_w.shape[0]-1, sharey=True)
    fig.suptitle(title)
    axs[0].set_ylabel(dist)
    axs_2 = [a.twinx() for a in axs]
    axs_2[-1].set_ylabel('|w|_1')
    for p in range(track_active_weights_w.shape[0]-1):
        axs[p].plot(track_distance[p, :].T, color='k', label=dist)
        axs[p].set_xlabel('Iterations')
        axs_2[p].plot(track_active_weights_w[p, :].T, label='view 1 weights', linestyle=':')
        axs_2[p].plot(track_active_weights_c[p, :].T, label='view 2 weights', linestyle=':')
        if dist == 'correlation':
            axs[p].set_ylim(bottom=0, top=1.1)
            axs[p].set_xlim(left=0, right=100)
        else:
            axs[p].set_ylim(bottom=0, top=max_obj*1.1)
        axs_2[p].set_ylim(bottom=0, top=max_norm*1.1)
        axs_2[p].set_xlim(left=0, right=100)
        if p == track_active_weights_w.shape[0] - 1:
            axs[p].set_title('Normal ALS')
        else:
            axs[p].set_title('c={:.2E}'.format(params[p]))
        handles, labels = axs_2[p].get_legend_handles_labels()
        handles2, labels2 = axs[p].get_legend_handles_labels()
        handles = handles + handles2
        labels = labels + labels2
    fig.legend(handles, labels,loc='lower center',ncol=3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88,bottom=0.2)
    plt.savefig("sparse_convergence/"+title+"_"+dist)


def convergence_test(X, Y, max_iters=1, parameter_grid=None, method='l2'):
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
        params = {'c_1': parameter_grid[l, 0], 'c_2': parameter_grid[l, 1], 'l1_ratio_1': 0.5, 'l1_ratio_2': 0.5}
        # This uses my ALS_inner_loop class
        ALS = CCA_methods.ALS_inner_loop(X, Y, method=method, params=params, max_iter=max_iters)
        # Get the statistics
        track_correlations[l, :] = ALS.track_correlation
        track_norms_w[l, :] = ALS.track_norms_w
        track_norms_c[l, :] = ALS.track_norms_c
        # Useful for the John one
        track_obj[l, :] = ALS.track_obj
        w[l, :] = ALS.w
        c[l, :] = ALS.c

    # Run an ALS CCA for comparison
    ALS = CCA_methods.alternating_least_squares.ALS_inner_loop(X, Y, method='l2', params=None, max_iter=max_iters)
    track_correlations[-1, :] = ALS.track_correlation
    track_norms_w[-1, :] = ALS.track_norms_w
    track_norms_c[-1, :] = ALS.track_norms_c
    # track_obj /= np.nanmax(track_obj)
    return track_correlations, track_norms_w, track_norms_c, track_obj

max_iters=100
X, Y, _,_ = generate_mai(100, 1, 100, 100, 10, 10)
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)

number_of_parameters_to_try = 5

### SCCA vs. SGCCA

parameter_grid = np.ones((number_of_parameters_to_try, 2)) * 0
parameter_grid[:, 0] = np.linspace(2e-2, 1e-3, number_of_parameters_to_try)
parameter_grid[:, 1] = np.linspace(2e-2, 1e-3, number_of_parameters_to_try)

scca = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='scca')
sgcca = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='sgcca')

max_norm = max(np.nanmax(scca[1]), np.nanmax(sgcca[1]),
               np.nanmax(scca[2]), np.nanmax(sgcca[2]))

max_obj = max(np.nanmax(scca[3]), np.nanmax(sgcca[3]))

dist_spar_plot(scca[3], scca[1], scca[2], parameter_grid[:, 0], 'SCCA', dist='Objective Function',
               max_norm=max_norm,max_obj=max_obj)

dist_spar_plot(scca[0], scca[1], scca[2], parameter_grid[:, 0], 'SCCA',
               max_norm=max_norm)

dist_spar_plot(sgcca[3], sgcca[1], sgcca[2], parameter_grid[:, 0], 'SGCCA',
               dist='Objective Function',
               max_norm=max_norm,max_obj=max_obj)

### Constrained SCCA

parameter_grid = np.ones((number_of_parameters_to_try, 2)) * 0
parameter_grid[:, 0] = np.linspace(0.1, 0.5, number_of_parameters_to_try)
parameter_grid[:, 1] = np.linspace(0.1, 0.5, number_of_parameters_to_try)

constrained_scca = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='constrained_scca')

dist_spar_plot(constrained_scca[0], constrained_scca[1], constrained_scca[2], parameter_grid[:, 0], 'Constrained SCCA',
               max_norm=max_norm)

# PMD

parameter_grid = np.ones((number_of_parameters_to_try, 2)) * 0
parameter_grid[:, 0] = np.linspace(10, 1, number_of_parameters_to_try)
parameter_grid[:, 1] = np.linspace(10, 1, number_of_parameters_to_try)

pmd = convergence_test(X, Y, max_iters=max_iters, parameter_grid=parameter_grid, method='pmd')

max_norm = max(np.nanmax(pmd[1]), np.nanmax(pmd[2]))

dist_spar_plot(pmd[0], pmd[1], pmd[2], parameter_grid[:, 0], 'PMD',
               max_norm=max_norm)

plt.show()

print('here')
