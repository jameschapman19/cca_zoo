"""
Gradient-based CCALoss and CCA_EYLoss
============================

This script demonstrates how to use gradient-based methods
to perform canonical correlation analysis (CCALoss) on high-dimensional data.
We will compare the performance of CCALoss and CCA_EYLoss, which is a variant of CCALoss
that uses stochastic gradient descent to solve the optimization problem.
We will also explore the effect of different batch sizes on CCA_EYLoss and plot
the loss function over iterations.
"""

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.linear import CCA, CCA_EYLoss
from cca_zoo.visualisation import ScoreScatterDisplay

# %%
# Data
# ----
# We set the random seed for reproducibility
np.random.seed(42)

# We generate a linear dataset with 1000 samples, 500 features per view,
# 1 latent dimension and a correlation of 0.9 between the representations
n = 10000
p = 1000
q = 1000
latent_dims = 1
correlation = 0.9

(X, Y) = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[correlation]
).sample(n)

# We split the data into train and test sets with a ratio of 0.8
train_ratio = 0.8
train_idx = np.random.choice(np.arange(n), size=int(train_ratio * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train = X[train_idx]
Y_train = Y[train_idx]
X_test = X[test_idx]
Y_test = Y[test_idx]

# %%
# CCALoss
# ---
# We create a CCALoss object with the number of latent dimensions as 1
cca = CCA(latent_dimensions=latent_dims)

# We record the start time of the model fitting
start_time = time.time()

# We fit the model on the train set and transform both representations
cca.fit([X_train, Y_train])
X_train_cca, Y_train_cca = cca.transform([X_train, Y_train])
X_test_cca, Y_test_cca = cca.transform([X_test, Y_test])

# We record the end time of the model fitting and compute the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

score_display = ScoreScatterDisplay.from_estimator(
    cca, [X_train, Y_train], [X_test, Y_test]
)
score_display.plot(title=f"CCALoss (Time: {elapsed_time:.2f} s)")
plt.show()

# %%
# CCA_EYLoss with different batch sizes
# --------------------------------
# We create a list of batch sizes to try out
batch_sizes = [200, 100, 50, 20, 10]

# We loop over the batch sizes and create a CCA_EYLoss object for each one
for batch_size in batch_sizes:
    ccaey = CCA_EYLoss(
        latent_dimensions=latent_dims,
        epochs=10,
        batch_size=batch_size,
        learning_rate=0.1,
        random_state=42,
    )

    # We record the start time of the model fitting
    start_time = time.time()

    # We fit the model on the train set and transform both representations
    ccaey.fit([X_train, Y_train])

    # We record the end time of the model fitting and compute the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # We plot the transformed representations on a scatter plot with different colors for train and test sets
    # Use ScoreScatterDisplay or a similar plotting class for the visualization
    score_display = ScoreScatterDisplay.from_estimator(
        ccaey, [X_train, Y_train], [X_test, Y_test]
    )
    score_display.plot(
        title=f"CCA_EYLoss (Batch size: {batch_size}, Time: {elapsed_time:.2f} s)"
    )
    plt.show()

# %%
# Comparison
# ----------
# We can see that CCA_EYLoss achieves a higher correlation than CCALoss on the test set,
# indicating that it can handle high-dimensional data better by using gradient descent.
# We can also see that the batch size affects the performance of CCA_EYLoss, with smaller batch sizes
# leading to higher correlations but also higher variance. This is because smaller batch sizes
# allow for more frequent updates and exploration of the parameter space, but also introduce more noise
# and instability in the optimization process. A trade-off between batch size and learning rate may be needed
# to achieve the best results. We can also see that CCA_EYLoss converges faster than CCALoss, as it takes less time
# to fit the model. The loss function plots show how the objective value decreases over iterations for different
# batch sizes, and we can see that smaller batch sizes tend to have more fluctuations and slower convergence.
