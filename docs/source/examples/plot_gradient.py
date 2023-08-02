"""
Gradient-based CCA and CCAEY
============================

This script demonstrates how to use gradient-based methods
to perform canonical correlation analysis (CCA) on high-dimensional data.
We will compare the performance of CCA and CCAEY, which is a variant of CCA
that uses stochastic gradient descent to solve the optimization problem.
We will also explore the effect of different batch sizes on CCAEY and plot
the loss function over iterations.
"""

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.linear import CCA, CCAEY

# %%
# Data
# ----
# We set the random seed for reproducibility
np.random.seed(42)

# We generate a linear dataset with 1000 samples, 500 features per view,
# 1 latent dimension and a correlation of 0.9 between the views
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
# CCA
# ---
# We create a CCA object with the number of latent dimensions as 1
cca = CCA(latent_dimensions=latent_dims)

# We record the start time of the model fitting
start_time = time.time()

# We fit the model on the train set and transform both views
cca.fit([X_train, Y_train])
X_train_cca, Y_train_cca = cca.transform([X_train, Y_train])
X_test_cca, Y_test_cca = cca.transform([X_test, Y_test])

# We record the end time of the model fitting and compute the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# We compute the correlation between the transformed views on the test set
cca_corr = np.corrcoef(X_test_cca[:, 0], Y_test_cca[:, 0])[0, 1]

# We plot the transformed views on a scatter plot with different colors for train and test sets
plt.figure()
plt.scatter(X_train_cca[:, 0], Y_train_cca[:, 0], c="b", label="Train")
plt.scatter(X_test_cca[:, 0], Y_test_cca[:, 0], c="r", label="Test")
plt.xlabel("views latent")
plt.ylabel("Y latent")
plt.title(f"CCA (Corr: {cca_corr:.2f}, Time: {elapsed_time:.2f} s)")
plt.legend()
plt.show()

# %%
# CCAEY with different batch sizes
# --------------------------------
# We create a list of batch sizes to try out
batch_sizes = [200, 100, 50, 20, 10]

# We loop over the batch sizes and create a CCAEY object for each one
for batch_size in batch_sizes:
    ccaey = CCAEY(
        latent_dimensions=latent_dims,
        epochs=5,
        batch_size=batch_size,
        learning_rate=0.001,
        random_state=42,
        track="loss",
    )

    # We record the start time of the model fitting
    start_time = time.time()

    # We fit the model on the train set and transform both views
    ccaey.fit([X_train, Y_train])
    X_train_ccae, Y_train_ccae = ccaey.transform([X_train, Y_train])
    X_test_ccae, Y_test_ccae = ccaey.transform([X_test, Y_test])

    # We record the end time of the model fitting and compute the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # We compute the correlation between the transformed views on the test set
    ccaey_corr = np.corrcoef(X_test_ccae[:, 0], Y_test_ccae[:, 0])[0, 1]

    # We plot the transformed views on a scatter plot with different colors for train and test sets
    plt.figure()
    plt.scatter(X_train_ccae[:, 0], Y_train_ccae[:, 0], c="b", label="Train")
    plt.scatter(X_test_ccae[:, 0], Y_test_ccae[:, 0], c="r", label="Test")
    plt.xlabel("views latent")
    plt.ylabel("Y latent")
    plt.title(
        f"CCAEY (Batch size: {batch_size}, Corr: {ccaey_corr:.2f}, Time: {elapsed_time:.2f} s)"
    )
    plt.legend()
    plt.show()

    # We plot the loss function over iterations
    plt.figure()
    plt.plot(ccaey.objective)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"CCAEY (Batch size: {batch_size})")
    plt.show()

# %%
# Comparison
# ----------
# We can see that CCAEY achieves a higher correlation than CCA on the test set,
# indicating that it can handle high-dimensional data better by using gradient descent.
# We can also see that the batch size affects the performance of CCAEY, with smaller batch sizes
# leading to higher correlations but also higher variance. This is because smaller batch sizes
# allow for more frequent updates and exploration of the parameter space, but also introduce more noise
# and instability in the optimization process. A trade-off between batch size and learning rate may be needed
# to achieve the best results. We can also see that CCAEY converges faster than CCA, as it takes less time
# to fit the model. The loss function plots show how the objective value decreases over iterations for different
# batch sizes, and we can see that smaller batch sizes tend to have more fluctuations and slower convergence.
