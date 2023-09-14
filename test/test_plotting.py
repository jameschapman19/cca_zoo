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

X = np.random.rand(100, 10)
Y = np.random.rand(100, 10)
Z = np.random.rand(100, 10)

# Train Test Split
X_train, X_test = X[:50], X[50:]
Y_train, Y_test = Y[:50], Y[50:]
Z_train, Z_test = Z[:50], Z[50:]

views = [X_train, Y_train, Z_train]
test_views = [X_test, Y_test, Z_test]

# MCCA
mcca = MCCA(latent_dimensions=2)
mcca.fit(views)

# Explained Variance
ExplainedVarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# Explained Covariance
ExplainedCovarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# Weight heatmap
WeightHeatmapDisplay.from_estimator(mcca).plot()

# Score heatmap
ScoreDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# Covariance heatmap
CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()

# Correlation heatmap
CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()

plt.show()
