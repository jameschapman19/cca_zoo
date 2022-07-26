import numpy as np
from cca_zoo.data import generate_covariance_data
from sklearn.model_selection import train_test_split

from ._utils import demean, scale


def linear_dataset(components, model="cca", random_state=0):
    N = 10000
    rng = np.random.default_rng(random_state)
    if model == "cca":
        (X, Y), _ = generate_covariance_data(
            N,
            [50, 50],
            latent_dims=components,
            correlation=list(np.linspace(0, 0.9, components + 1))[1:],
            structure="toeplitz",
            sigma=0.5,
            random_state=random_state,
            view_sparsity=0.1
        )

    else:
        k = np.linspace(0, 100, 50 + 1)
        Z = np.linalg.qr(rng.standard_normal(size=(N, components)))[0] * k[1:]
        X = Z @ np.linalg.pinv(
            np.linalg.qr(rng.standard_normal(size=(components, 50)))[0]
        )
        Y = Z @ np.linalg.pinv(
            np.linalg.qr(rng.standard_normal(size=(components, 50)))[0]
        )
    X, X_te, Y, Y_te = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    X, X_te, Y, Y_te = scale(*demean(X, X_te, Y, Y_te))
    return X, Y, X_te, Y_te


def main():
    X, Y, X_te, Y_te = linear_dataset()  # S*1
    _, S, _ = np.linalg.svd(X.T @ Y)
    print()


if __name__ == "__main__":
    main()
