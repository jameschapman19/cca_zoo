"""
Visualizing CCALoss Models with CCALoss-Zoo
====================================

Ever wondered how to peek into the inner workings of your Canonical Correlation Analysis (CCALoss) models?
This example will guide you through CCALoss-Zoo's built-in plotting functionalities, showing you the keys to unlock those insights!
"""

# %%
# Import the Essentials
# ----------------------
from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    ScoreScatterDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
    SeparateScoreScatterDisplay,
    JointScoreScatterDisplay,
    SeparateJointScoreDisplay,
    PairScoreScatterDisplay,
)
import matplotlib.pyplot as plt
import numpy as np
from cca_zoo.linear import MCCA

# %%
# Cooking Up Some Data
# --------------------
# We create synthetic data for three different representations, which we'll use for training and testing our model.
X = np.random.rand(100, 10)
Y = np.random.rand(100, 10)
Cats = np.random.randint(0, 2, 100)

# Presto! Splitting the data into training and testing sets.
X_train, X_test = X[:50], X[50:]
Y_train, Y_test = Y[:50], Y[50:]
Cats_train, Cats_test = Cats[:50], Cats[50:]

views = [X_train, Y_train]
test_views = [X_test, Y_test]

# %%
# The Training Ritual
# -------------------
# We'll use Multi-Set Canonical Correlation Analysis (MCCALoss) to find shared patterns among the three representations.
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
# Explained covariance dives deeper, revealing how well your model explains the covariance structure between different representations.
ExplainedCovarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()
plt.show()
print(
    "Hint: The closer to one, the better your model captures the relation between representations."
)

# %%
# Peering into the Weights
# ------------------------
# Ever wondered how much each feature contributes? The weight heatmap unveils the importance of each feature in your model.
WeightHeatmapDisplay.from_estimator(mcca).plot()
plt.show()
print("Did you know? Large weights_ are usually more influential in the model.")

# The Scoreboard
# --------------
# Score heatmaps help you visualize how the CCALoss projections from multiple representations relate to each other.

# Example using ScoreScatterDisplay
score_plot = ScoreScatterDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
)
score_plot.plot()
plt.show()
print(
    "In this plot, you can visualize the CCALoss projections from multiple representations. It's useful for identifying clusters or patterns, which can help validate your model's effectiveness."
)

# Example using SeparateScoreScatterDisplay
separate_score_plot = SeparateScoreScatterDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
)
separate_score_plot.plot()
plt.show()
print(
    "This plot separates the train and test scores onto different figures. It's helpful when you want to compare the performance of your model on training and testing data separately."
)

# Example using JointScoreScatterDisplay
joint_score_plot = JointScoreScatterDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
)
joint_score_plot.plot()
plt.show()
print(
    "The Joint Plot shows the distribution of the scores for each view on the x and y axis. It can help you understand how the two representations relate to each other in a joint distribution."
)

# Example using SeparateJointScoreDisplay
separate_joint_score_plot = SeparateJointScoreDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
)
separate_joint_score_plot.plot()
plt.show()
print(
    "Similar to the Joint Plot, but it separates train and test scores onto different figures. Useful when you want to analyze their distributions separately."
)

# Example using PairScoreScatterDisplay
pair_score_plot = PairScoreScatterDisplay.from_estimator(
    mcca, views, test_views=test_views, labels=Cats_train, test_labels=Cats_test
)
pair_score_plot.plot()
plt.show()
print(
    "The Pair Plot visualizes the pairwise relationships between scores. It can be helpful for identifying correlations or dependencies between representations."
)

# %%
# The Covariance Matrix: A Mirror Into Your Model
# -----------------------------------------------
# The covariance heatmap provides a detailed look at how features from different representations covary.
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
