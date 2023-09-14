"""
Visualizing CCA Models with CCA-Zoo
====================================

This example demonstrates how to use CCA-Zoo's built-in plotting functionalities
to visualize Canonical Correlation Analysis (CCA) models.
"""

# %%
# Import Required Libraries
# -------------------------
from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    ScoreDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
)
import matplotlib.pyplot as plt
import numpy as np
from cca_zoo.linear import MCCA

# %%
# Generate Sample Data
# --------------------
X = np.random.rand(100, 10)
Y = np.random.rand(100, 10)
Z = np.random.rand(100, 10)

# Splitting the data into training and testing sets
X_train, X_test = X[:50], X[50:]
Y_train, Y_test = Y[:50], Y[50:]
Z_train, Z_test = Z[:50], Z[50:]

views = [X_train, Y_train, Z_train]
test_views = [X_test, Y_test, Z_test]

# %%
# Train an MCCA Model
# -------------------
mcca = MCCA(latent_dimensions=2)
mcca.fit(views)

# %%
# Plotting the Explained Variance
# -------------------------------
ExplainedVarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# %%
# Plotting the Explained Covariance
# ---------------------------------
ExplainedCovarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# %%
# Visualizing the Weights using a Heatmap
# ----------------------------------------
WeightHeatmapDisplay.from_estimator(mcca).plot()

# %%
# Score Heatmap
# -------------
ScoreDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# %%
# Covariance Heatmap
# ------------------
CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# %%
# Correlation Heatmap
# -------------------
CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# %%
# Show all plots
# --------------
plt.show()
