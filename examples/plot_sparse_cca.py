"""
Sparse CCA Methods
===========================

This script shows how regularised methods can be used to extract sparse solutions to the CCA problem
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.models import (
    SCCA_PMD,
    SCCA_IPLS,
    ElasticCCA,
    CCA,
    PLS,
    SCCA_Span,
    AltMaxVar,
)


def plot_true_weights_coloured(ax, weights, true_weights, title=""):
    ind = np.arange(len(true_weights))
    mask = np.squeeze(true_weights == 0)
    ax.scatter(ind[~mask], weights[~mask], c="b")
    ax.scatter(ind[mask], weights[mask], c="r")
    ax.set_title(title)


def plot_model_weights(wx, wy, tx, ty, title=""):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plot_true_weights_coloured(axs[0, 0], tx, tx, title="true x weights")
    plot_true_weights_coloured(axs[0, 1], ty, ty, title="true y weights")
    plot_true_weights_coloured(axs[1, 0], wx, tx, title="model x weights")
    plot_true_weights_coloured(axs[1, 1], wy, ty, title="model y weights")
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()


# %%
# Data
# -----
np.random.seed(42)
n = 200
p = 100
q = 100
view_1_sparsity = 0.1
view_2_sparsity = 0.1
latent_dims = 1

data = LinearSimulatedData(
    view_features=[p, q],
    latent_dims=latent_dims,
    view_sparsity=[view_1_sparsity, view_2_sparsity],
    correlation=[0.9],
)
(X, Y) = data.sample(n)

tx = data.true_features[0] / np.sqrt(n)
ty = data.true_features[1] / np.sqrt(n)

# %% CCA
cca = CCA().fit([X, Y])
plot_model_weights(cca.weights[0], cca.weights[1], tx, ty, title="CCA")

# %% PLS
pls = PLS().fit([X, Y])
plot_model_weights(pls.weights[0], pls.weights[1], tx, ty, title="PLS")

# %% PMD
tau1 = [0.1, 0.3, 0.7, 0.9]
tau2 = [0.1, 0.3, 0.7, 0.9]
param_grid = {"tau": [tau1, tau2]}
pmd = GridSearchCV(SCCA_PMD(), param_grid=param_grid, cv=3, verbose=True).fit([X, Y])

# %%
plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(pmd.best_estimator_.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")
# %%
plot_model_weights(
    pmd.best_estimator_.weights[0], pmd.best_estimator_.weights[1], tx, ty, title="PMD"
)

# %%
pd.DataFrame(pmd.cv_results_)

# %% IPLS
scca = SCCA_IPLS(tau=[1e-2, 1e-2]).fit([X, Y])
plot_model_weights(scca.weights[0], scca.weights[1], tx, ty, title="SCCA_IPLS")

plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(scca.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")

scca_pos = SCCA_IPLS(tau=[1e-2, 1e-2], positive=[True, True]).fit([X, Y])
plot_model_weights(
    scca_pos.weights[0], scca_pos.weights[1], tx, ty, title="SCCA_IPLS (Positive)"
)

plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(scca_pos.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")


elasticcca = ElasticCCA(alpha=[1e-2, 1e-2], l1_ratio=[0.5, 0.5]).fit([X, Y])
plot_model_weights(
    elasticcca.weights[0], elasticcca.weights[1], tx, ty, title="ELastic CCA"
)

plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(elasticcca.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")


altmaxvar = AltMaxVar(tau=[1e-5, 1e-5]).fit([X, Y])
plot_model_weights(
    altmaxvar.weights[0], altmaxvar.weights[1], tx, ty, title="AltMaxVar"
)

plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(altmaxvar.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")


spancca = SCCA_Span(tau=[10, 10], max_iter=2000, rank=20).fit([X, Y])
plot_model_weights(spancca.weights[0], spancca.weights[1], tx, ty, title="Span CCA")

plt.figure()
plt.title("Objective Convergence")
plt.plot(np.array(spancca.track["objective"][0]))
plt.ylabel("Objective")
plt.xlabel("#iterations")
