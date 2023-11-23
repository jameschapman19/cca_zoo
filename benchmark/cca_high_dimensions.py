"""
Benchmarking CCA on high dimensional data. Using CCA-Zoo and Scikit-learn.

Use different dimensionalities and produce a nice seaborn plot of the runtimes.
"""

import time
import pandas as pd
import numpy as np
from cca_zoo.linear import CCA
from sklearn.cross_decomposition import CCA as sklearn_CCA
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize empty list to hold the benchmarking results
results = []

# List of dimensions to test
dimensions = [50, 100, 200, 400, 800]

# Number of samples
n_samples = 100

# Latent dimension
latent_dimensions = 10

# Number of repeats
n_repeats = 10

# Loop over each dimensionality
for dim in dimensions:
    for repeat in range(n_repeats):
        # Generate synthetic data
        X = np.random.rand(n_samples, dim)
        Y = np.random.rand(n_samples, dim)

        # CCA-Zoo
        start_time = time.time()
        cca_zoo = CCA(latent_dimensions=latent_dimensions)
        cca_zoo.fit((X, Y))
        cca_zoo_time = time.time() - start_time

        # Record results
        results.append(
            {"Dimension": dim, "Time": cca_zoo_time, "Method": "CCA-Zoo"}
        )

        # Scikit-learn
        start_time = time.time()
        sk_cca = sklearn_CCA(n_components=latent_dimensions)
        sk_cca.fit(X, Y)
        sklearn_time = time.time() - start_time

        # Record results
        results.append(
            {"Dimension": dim, "Time": sklearn_time, "Method": "Scikit-learn"}
        )

# Convert to DataFrame
df = pd.DataFrame(results)

# Seaborn Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Dimension", y="Time", hue="Method", marker="o", errorbar="sd")
plt.title("CCA Performance comparison with Uncertainty")
plt.xlabel("Dimension")
plt.ylabel("Average Execution Time (seconds)")
plt.tight_layout()
plt.savefig("CCA_Speed_Benchmark.pdf")
plt.show()
