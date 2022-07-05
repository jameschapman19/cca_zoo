import numpy as np
from cca_zoo.data import generate_covariance_data
from sklearn.model_selection import train_test_split

from ._utils import demean


def exponential_dataset(components, model="cca", random_state=0):
    N = 1000
    rng = np.random.default_rng(random_state)
    if model == "cca":
        (X, Y), _ = generate_covariance_data(
            1000,
            [50, 50],
            latent_dims=components,
            correlation=0.99,
            decay=0.9,
            structure="toeplitz",
            sigma=0.5,
            random_state=random_state,
        )
    else:
        k = 10 ** np.linspace(0, 3, 50 + 1)
        Z = np.linalg.qr(rng.standard_normal(size=(N, components)))[0] * k[1:]
        X = Z @ np.linalg.pinv(
            np.linalg.qr(rng.standard_normal(size=(components, 50)))[0]
        )
        Y = Z @ np.linalg.pinv(
            np.linalg.qr(rng.standard_normal(size=(components, 50)))[0]
        )
    X, X_te, Y, Y_te = train_test_split(X, Y, test_size=0.2)
    X, X_te, Y, Y_te = demean(X, X_te, Y, Y_te)
    return X, Y, X_te, Y_te


def main():
    X, Y, X_te, Y_te = exponential_dataset()  # S*1
    _, S, _ = np.linalg.svd(X.T @ Y)
    print()


if __name__ == "__main__":
    main()
