"""
Sparse CCA Methods
===========================

This script shows how regularised methods can be used to extract sparse solutions to the CCA problem
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca_zoo.data import generate_covariance_data
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.models import PMD, SCCA, ElasticCCA, CCA, PLS, SCCA_ADMM, SpanCCA

# %%
np.random.seed(42)
n = 200
p = 100
q = 100
view_1_sparsity = 0.1
view_2_sparsity = 0.1
latent_dims = 1

(X, Y), (tx, ty) = generate_covariance_data(
    n,
    view_features=[p, q],
    latent_dims=latent_dims,
    view_sparsity=[view_1_sparsity, view_2_sparsity],
    correlation=[0.9],
)
tx /= np.sqrt(n)
ty /= np.sqrt(n)


# %%
def plot_true_weights_coloured(ax, weights, true_weights, title=""):
    ind = np.arange(len(true_weights))
    mask = np.squeeze(true_weights == 0)
    ax.scatter(ind[~mask], weights[~mask], c="b")
    ax.scatter(ind[mask], weights[mask], c="r")
    ax.set_title(title)


def plot_model_weights(wx, wy, tx, ty):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plot_true_weights_coloured(axs[0, 0], tx, tx, title="true x weights")
    plot_true_weights_coloured(axs[0, 1], ty, ty, title="true y weights")
    plot_true_weights_coloured(axs[1, 0], wx, tx, title="model x weights")
    plot_true_weights_coloured(axs[1, 1], wy, ty, title="model y weights")
    plt.tight_layout()
    plt.show()


# %%
cca = CCA().fit([X, Y])
plot_model_weights(cca.weights[0], cca.weights[1], tx, ty)

# %%
pls = PLS().fit([X, Y])
plot_model_weights(pls.weights[0], pls.weights[1], tx, ty)

# %%
pmd = PMD(c=[2, 2]).fit([X, Y])
plot_model_weights(pmd.weights[0], pmd.weights[1], tx, ty)

# %%
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(pmd.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")

# %%
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_grid = {"c": [c1, c2]}
pmd = GridSearchCV(PMD(), param_grid=param_grid, cv=3, verbose=True).fit([X, Y])

# %%
pd.DataFrame(pmd.cv_results_)

# %%
scca = SCCA(c=[1e-3, 1e-3]).fit([X, Y])
plot_model_weights(scca.weights[0], scca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(scca.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")

# %%
scca_pos = SCCA(c=[1e-3, 1e-3], positive=[True, True]).fit([X, Y])
plot_model_weights(scca_pos.weights[0], scca_pos.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(scca_pos.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")

# %%
elasticcca = ElasticCCA(c=[10000, 10000], l1_ratio=[0.000001, 0.000001]).fit([X, Y])
plot_model_weights(elasticcca.weights[0], elasticcca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(elasticcca.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")

# %%
scca_admm = SCCA_ADMM(c=[1e-3, 1e-3]).fit([X, Y])
plot_model_weights(scca_admm.weights[0], scca_admm.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(scca_admm.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")

# %%
spancca = SpanCCA(c=[10, 10], max_iter=2000, rank=20).fit([X, Y])
plot_model_weights(spancca.weights[0], spancca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(spancca.track[0]["objective"]).T)
plt.ylabel("Objective")
plt.xlabel("#iterations")
