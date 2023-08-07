"""
Sparse CCA Comparison
===========================

This example demonstrates how to easily train Sparse CCA variants
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cca_zoo.linear import (
    CCA,
    PLS,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    SCCA_Span,
)
from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV

# Setting plot style and font scale for better visibility
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.2)
plt.close("all")

# Function to create a scatterplot of weights
def plot_true_weights_coloured(ax, weights, true_weights, title="", legend=False):
    # Preprocess weights for seaborn scatterplot
    weights = np.squeeze(weights)
    ind = np.arange(len(true_weights))
    mask = np.squeeze(true_weights == 0)

    non_zero_df = pd.DataFrame({
        'Index': ind[~mask],
        'Weights': weights[~mask],
        'Type': 'Non-Zero Weights'
    })
    zero_df = pd.DataFrame({
        'Index': ind[mask],
        'Weights': weights[mask],
        'Type': 'Zero Weights'
    })
    data_df = pd.concat([non_zero_df, zero_df])

    # Create seaborn scatterplot
    sns.scatterplot(data=data_df, x='Index', y='Weights', hue='Type', ax=ax, palette="viridis", legend=legend)
    ax.set_title(title)

# Function to plot weights for each model
def plot_model_weights(wx, wy, tx, ty, title=""):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    plot_true_weights_coloured(axs[0, 0], tx, tx, title="true x weights")
    plot_true_weights_coloured(axs[0, 1], ty, ty, title="true y weights")
    plot_true_weights_coloured(axs[1, 0], wx, tx, title="model x weights")
    plot_true_weights_coloured(axs[1, 1], wy, ty, title="model y weights", legend=True)

    # Add legend to the plot
    handles, labels = axs[1, 1].get_legend_handles_labels()
    # legend off
    plt.legend([], [], frameon=False)
    fig.legend(handles, labels,bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=2)
    plt.tight_layout()
    fig.suptitle(title)
    sns.despine(trim=True)
    plt.show(block=False)

# Initialize parameters
np.random.seed(42)
n = 500
p = 200
q = 200
view_1_sparsity = 0.1
view_2_sparsity = 0.1
latent_dims = 1
epochs = 50

# Simulate some data
data = LinearSimulatedData(
    view_features=[p, q],
    latent_dims=latent_dims,
    view_sparsity=[view_1_sparsity, view_2_sparsity],
    correlation=[0.9],
)
(X, Y) = data.sample(n)

tx = data.true_features[0]
ty = data.true_features[1]

# Split data into train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

# Define a helper function to train and evaluate a model
def train_and_evaluate(model, title):
    model.fit([X_train, Y_train])
    plot_model_weights(model.weights[0], model.weights[1], tx, ty, title=title)
    return model.score([X_val, Y_val])

# Train and evaluate each model
cca_corr = train_and_evaluate(CCA(), "CCA")
pls_corr = train_and_evaluate(PLS(), "PLS")
span_cca_corr = train_and_evaluate(SCCA_Span(tau=[10, 10], early_stopping=True), "Span CCA")

# For PMD model we use GridSearchCV
tau1 = [0.1, 0.5, 0.9]
tau2 = [0.1, 0.5, 0.9]
param_grid = {"tau": [tau1, tau2]}
pmd = GridSearchCV(SCCA_PMD(epochs=epochs, early_stopping=True), param_grid=param_grid).fit([X_train, Y_train])
plot_model_weights(pmd.best_estimator_.weights[0], pmd.best_estimator_.weights[1], tx, ty, title="PMD")
pmd_corr = pmd.score([X_val, Y_val])

# Training and evaluating SCCA_IPLS models
scca_corr = train_and_evaluate(SCCA_IPLS(alpha=[1e-2, 1e-2], epochs=epochs, early_stopping=True), "SCCA_IPLS")
scca_pos_corr = train_and_evaluate(SCCA_IPLS(alpha=[1e-2, 1e-2], positive=True, epochs=epochs, early_stopping=True), "SCCA_IPLS+")


# Elastic CCA Model
alpha = [1e-2,1e-3]
param_grid = {"alpha": alpha}
elastic = GridSearchCV(ElasticCCA(epochs=epochs, early_stopping=True, l1_ratio=0.99), param_grid=param_grid).fit([X_train, Y_train])
plot_model_weights(elastic.best_estimator_.weights[0], elastic.best_estimator_.weights[1], tx, ty, title="ElasticCCA")
elastic_corr = elastic.score([X_val, Y_val])

# Print final comparison
print("CCA Correlation: ", cca_corr)
print("PLS Correlation: ", pls_corr)
print("Span CCA Correlation: ", span_cca_corr)
print("PMD Correlation: ", pmd_corr)
print("SCCA_IPLS Correlation: ", scca_corr)
print("SCCA_IPLS+ Correlation: ", scca_pos_corr)
print("ElasticCCA Correlation: ", elastic_corr)

# Store model names and correlations in a dictionary
model_results = {
    "CCA": cca_corr,
    "PLS": pls_corr,
    "Span CCA": span_cca_corr,
    "PMD": pmd_corr,
    "SCCA_IPLS": scca_corr,
    "SCCA_IPLS (Positive)": scca_pos_corr,
    "Elastic CCA": elastic_corr,
}

# Convert dictionary to pandas DataFrame for easy plotting
results_df = pd.DataFrame.from_dict(model_results, orient='index', columns=['Validation Correlation'])

# Plot the data
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df['Validation Correlation'], palette="viridis")
plt.xticks(rotation=90)
plt.title("Comparison of Models by Validation Correlation")
plt.ylabel("Validation Correlation")
plt.tight_layout()
plt.show()
