"""
Visualizing CCA Models with CCA-Zoo
====================================

Ever wondered how to peek into the inner workings of your Canonical Correlation Analysis (CCA) models?
This example will guide you through CCA-Zoo's built-in plotting functionalities, showing you the keys to unlock those insights!
"""

# %%
# Import the Essentials
# ----------------------
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
# Cooking Up Some Data
# --------------------
# We create synthetic data for three different views, which we'll use for training and testing our model.
X = np.random.rand(100, 10)
Y = np.random.rand(100, 10)
Z = np.random.rand(100, 10)
Cats = np.random.randint(0, 2, 100)

# Presto! Splitting the data into training and testing sets.
X_train, X_test = X[:50], X[50:]
Y_train, Y_test = Y[:50], Y[50:]
Z_train, Z_test = Z[:50], Z[50:]
Cats_train, Cats_test = Cats[:50], Cats[50:]

views = [X_train, Y_train, Z_train]
test_views = [X_test, Y_test, Z_test]

# %%
# The Training Ritual
# -------------------
# We'll use Multi-Set Canonical Correlation Analysis (MCCA) to find shared patterns among the three views.
mcca = MCCA(latent_dimensions=2)
mcca.fit(views)

# %%
# Why So Varied? Understanding Explained Variance
# ----------------------------------------------
# Explained variance can give you a quick insight into how well your model captures the variance in each view.
ExplainedVarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()
plt.show()
print("Tip: Higher explained variance usually indicates better model fit.")

# %%
# When Covariance is Not Covert
# -----------------------------
# Explained covariance dives deeper, revealing how well your model explains the covariance structure between different views.
ExplainedCovarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()
plt.show()
print(
    "Hint: The closer to one, the better your model captures the relation between views."
)

# %%
# Peering into the Weights
# ------------------------
# Ever wondered how much each feature contributes? The weight heatmap unveils the importance of each feature in your model.
WeightHeatmapDisplay.from_estimator(mcca).plot()
plt.show()
print("Did you know? Large weights are usually more influential in the model.")

# %%
# The Scoreboard
# --------------
# Score heatmaps help you visualize how the CCA projections from multiple views relate to each other.
ScoreDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
).plot()
plt.show()
print(
    "Takeaway: Looking for clusters or patterns here can validate your model's effectiveness."
)

# %%
# The Covariance Matrix: A Mirror Into Your Model
# -----------------------------------------------
# The covariance heatmap provides a detailed look at how features from different views covary.
CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
plt.show()
print(
    "Keep an eye on: Strong covariances can hint at underlying patterns in your data."
)

# %%
# The Correlation Heatmap: A Normalized Tale
# ------------------------------------------
# This heatmap normalizes the covariance, giving you a measure that's easier to compare across different scales.
CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
plt.show()
print("Remember: Correlation values range from -1 to 1, making them easy to interpret.")
