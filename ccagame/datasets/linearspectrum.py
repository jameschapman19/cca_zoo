from jax._src import random
import numpy as np
from cca_zoo.data import generate_covariance_data
from sklearn.model_selection import train_test_split


def linear_dataset(cca=False, random_state=0):
    N=1000
    if cca:
        (X, Y), _ = generate_covariance_data(
            N,
            [50, 50],
            latent_dims=16,
            correlation=list(np.linspace(0, 1, 16)),
            structure="toeplitz",
            sigma=0.5,
            random_state=random_state,
        )
    else:
        (X, Y), _ = generate_covariance_data(
            N,
            [50, 50],
            latent_dims=16,
            decay=0.95,
            random_state=random_state,
        )
    X, X_te, Y, Y_te = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    return X, Y, X_te, Y_te
