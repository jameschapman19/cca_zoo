"""
Sparse CCA Variants Comparison
==============================

This script illustrates the training and evaluation of various Sparse Canonical Correlation Analysis (CCA) variants using synthetic data.
For each variant, model weights are visualized, and their performance is compared based on their correlation score on validation data.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cca_zoo.datasets import JointData
from cca_zoo.linear import (
    CCA,
    PLS,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    SCCA_Span,
)
from cca_zoo.model_selection import GridSearchCV

# Plotting Configuration
# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set_style("whitegrid")
sns.set_context(
    "paper",
    font_scale=2.0,
    rc={"lines.linewidth": 2.5},
)
plt.close("all")


def plot_true_weights_coloured(ax, weights, true_weights, title="", legend=False):
    """
    Create a scatterplot of weights_ differentiating between zero and non-zero true weights.
    """
    weights = np.squeeze(weights)
    ind = np.arange(len(true_weights))
    mask = np.squeeze(true_weights == 0)

    non_zero_df = pd.DataFrame(
        {"Index": ind[~mask], "Weights": weights[~mask], "Type": "Non-Zero Weights"}
    )
    zero_df = pd.DataFrame(
        {"Index": ind[mask], "Weights": weights[mask], "Type": "Zero Weights"}
    )
    data_df = pd.concat([non_zero_df, zero_df])

    sns.scatterplot(
        data=data_df,
        x="Index",
        y="Weights",
        hue="Type",
        ax=ax,
        palette="viridis",
        legend=legend,
    )
    ax.set_title(title)


def plot_model_weights(wx, wy, tx, ty, title="", save_path=None):
    """
    Plot weights of the model against the true weights.
    """
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plot_true_weights_coloured(axs[0, 0], tx, tx, title="True x weights")
    plot_true_weights_coloured(axs[0, 1], ty, ty, title="True y weights")
    plot_true_weights_coloured(axs[1, 0], wx, tx, title="Model x weights")
    plot_true_weights_coloured(axs[1, 1], wy, ty, title="Model y weights", legend=True)
    plt.legend([], [], frameon=False)
    fig.suptitle(title)
    sns.despine(trim=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show(block=False)


# Data Generation
np.random.seed(42)
n, p, q = 500, 200, 200
latent_dims = 1
view_1_sparsity, view_2_sparsity = 0.1, 0.1
data = JointData(
    view_features=[p, q],
    latent_dims=latent_dims,
    view_sparsity=[view_1_sparsity, view_2_sparsity],
    correlation=[0.9],
    positive=True,
)
(X, Y) = data.sample(n)
tx, ty = data.true_features[0], data.true_features[1]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)


def train_and_evaluate(model, title):
    """
    Helper function to train, evaluate, and visualize weights of a model.
    """
    model.fit([X_train, Y_train])
    plot_model_weights(model.weights_[0], model.weights_[1], tx, ty, title=title)
    return model.score([X_val, Y_val])


# Model Training and Evaluation
epochs = 50
cca_corr = train_and_evaluate(CCA(), "CCA")
pls_corr = train_and_evaluate(PLS(), "PLS")
span_cca_corr = train_and_evaluate(
    SCCA_Span(tau=[10, 10], early_stopping=True), "Span CCA"
)
scca_corr = train_and_evaluate(
    SCCA_IPLS(alpha=[1e-2, 1e-2], epochs=epochs, early_stopping=True), "SCCA_IPLS"
)
scca_pos_corr = train_and_evaluate(
    SCCA_IPLS(alpha=[1e-2, 1e-2], positive=True, epochs=epochs, early_stopping=True),
    "SCCA_IPLS+",
)

# Grid Search for ElasticCCA and PMD models
param_grid_pmd = {"tau": [[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]]}
pmd = GridSearchCV(
    SCCA_PMD(epochs=epochs, early_stopping=True), param_grid=param_grid_pmd
).fit([X_train, Y_train])
pmd_corr = pmd.score([X_val, Y_val])
plot_model_weights(
    pmd.best_estimator_.weights_[0],
    pmd.best_estimator_.weights_[1],
    tx,
    ty,
    title="PMD",
)

param_grid_elastic = {"alpha": [1e-2, 1e-3]}
elastic = GridSearchCV(
    ElasticCCA(epochs=epochs, early_stopping=True, l1_ratio=0.99),
    param_grid=param_grid_elastic,
).fit([X_train, Y_train])
elastic_corr = elastic.score([X_val, Y_val])
plot_model_weights(
    elastic.best_estimator_.weights_[0],
    elastic.best_estimator_.weights_[1],
    tx,
    ty,
    title="ElasticCCA",
)

# Results Visualization
results_df = pd.DataFrame(
    {
        "Model": [
            "CCA",
            "PLS",
            "Span CCA",
            "PMD",
            "SCCA_IPLS",
            "SCCA_IPLS (Positive)",
            "Elastic CCA",
        ],
        "Validation Correlation": [
            cca_corr.item(),
            pls_corr.item(),
            span_cca_corr.item(),
            pmd_corr.item(),
            scca_corr.item(),
            scca_pos_corr.item(),
            elastic_corr.item(),
        ],
    }
)

plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="Validation Correlation", data=results_df, palette="viridis")
plt.xticks(rotation=90)
plt.title("Comparison of Models by Validation Correlation")
plt.tight_layout()
plt.show()
