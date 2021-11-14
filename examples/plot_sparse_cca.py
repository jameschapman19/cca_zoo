"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

# %%

# A tutorial on using cca-zoo to generate multiview models with sparsity on weights


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
true_latent_dims = 1

(X, Y), (tx, ty) = generate_covariance_data(n, view_features=[p, q], latent_dims=true_latent_dims,
                                            view_sparsity=[view_1_sparsity, view_2_sparsity], correlation=[0.9])
# normalize weights for comparability
tx /= np.sqrt(n)
ty /= np.sqrt(n)


# %%

def plot_true_weights_coloured(ax, weights, true_weights, title=''):
    ind = np.arange(len(true_weights))
    mask = np.squeeze(true_weights == 0)
    ax.scatter(ind[~mask], weights[~mask], c='b')
    ax.scatter(ind[mask], weights[mask], c='r')
    ax.set_title(title)


def plot_model_weights(wx, wy, tx, ty):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plot_true_weights_coloured(axs[0, 0], tx, tx, title='true x weights')
    plot_true_weights_coloured(axs[0, 1], ty, ty, title='true y weights')
    plot_true_weights_coloured(axs[1, 0], wx, tx, title='model x weights')
    plot_true_weights_coloured(axs[1, 1], wy, ty, title='model y weights')
    plt.tight_layout()
    plt.show()


# %%

## First try with CCA

# %%

# fit a cca model
cca = CCA().fit([X, Y])

plot_model_weights(cca.weights[0], cca.weights[1], tx, ty)

# %%

## PLS

# %%

# fit a pls model
pls = PLS().fit([X, Y])

plot_model_weights(pls.weights[0], pls.weights[1], tx, ty)

# %%

## Penalized Matrix Decomposition (Sparse CCA by Witten)
# Initially set c=2 for both views arbitrarily

# %%

# fit a pmd model
pmd = PMD(c=[2, 2]).fit([X, Y])

plot_model_weights(pmd.weights[0], pmd.weights[1], tx, ty)

# %%

## Tracking the objective
# For these iterative algorithms, you can access the convergence over iterations

# %%

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(pmd.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%

### We can also tune the hyperparameter using GridSearchCV

# %%

# Set up a grid. We can't use c<1 or c>sqrt(#features)
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_grid = {'c': [c1, c2]}

# GridSearchCV can use multiple cores (jobs) and takes folds (number of cv folds) as a parameter. It can also produce a plot.
pmd = GridSearchCV(PMD(), param_grid=param_grid,
                   cv=3,
                   verbose=True).fit([X, Y])

# %%

# Also the model object now has a pandas dataframe containing the results from each fold

# %%

pd.DataFrame(pmd.cv_results_)

# %%

## Sparse CCA by iterative lasso (Mai)

# %%

# fit a scca model
scca = SCCA(c=[1e-3, 1e-3]).fit([X, Y])

plot_model_weights(scca.weights[0], scca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(scca.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%

### Positivity Constraints
# In this case it isn't helpful (the data were generated with positive and negative weights) but is a cool functionality!

# %%

# fit a scca model with positivity constraints
scca_pos = SCCA(c=[1e-3, 1e-3], positive=[True, True]).fit([X, Y])

plot_model_weights(scca_pos.weights[0], scca_pos.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(scca_pos.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%

## Sparse CCA by iterative elastic net (adapted from Waaijenborg)

# %%

# fit an elastic model
# for some reason this model needs REALLY big l2 regularisation. This is actually
# the same level of l1 regularisation as SCCA
elasticcca = ElasticCCA(c=[10000, 10000], l1_ratio=[0.000001, 0.000001]).fit([X, Y])

plot_model_weights(elasticcca.weights[0], elasticcca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(elasticcca.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%

## Sparse CCA by ADMM

# %%

# fit a scca_admm model
scca_admm = SCCA_ADMM(c=[1e-3, 1e-3]).fit([X, Y])

plot_model_weights(scca_admm.weights[0], scca_admm.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(scca_admm.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%

## Sparse CCA by random projection (Span CCA)
# This time the regularisation parameter c is the l0 norm of the weights i.e. the maximum number of non-zero weights. Let's cheat and give it the correct numbers. We can also change the rank of the estimation as described in the paper

# %%

# fit a spancca model
spancca = SpanCCA(c=[10, 10], max_iter=2000, rank=20).fit([X, Y])

plot_model_weights(spancca.weights[0], spancca.weights[1], tx, ty)

# Convergence
plt.figure()
plt.title('Objective Convergence')
plt.plot(np.array(spancca.track[0]['objective']).T)
plt.ylabel('Objective')
plt.xlabel('#iterations')

# %%
