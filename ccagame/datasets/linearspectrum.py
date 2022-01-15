import numpy as np
from cca_zoo.data import generate_covariance_data
from jax._src import random
from sklearn.model_selection import train_test_split


def linear_dataset(cca=False, random_state=0):
    N = 1000
    COMPONENTS = 50
    rng=np.random.default_rng(random_state)
    if cca:
        (X, Y), _ = generate_covariance_data(
            N,
            [COMPONENTS, COMPONENTS],
            latent_dims=COMPONENTS,
            correlation=list(np.linspace(0, 1, COMPONENTS)),
            structure="toeplitz",
            sigma=0.5,
            random_state=random_state,
        )
    else:
        k = np.linspace(0, 100, COMPONENTS + 1)
        Z = np.linalg.qr(rng.standard_normal(size=(N, COMPONENTS)))[0] * k[1:]
        X = Z @ np.linalg.pinv(np.linalg.qr(rng.standard_normal(size=(COMPONENTS, COMPONENTS)))[0])
        Y = Z @ np.linalg.pinv(np.linalg.qr(rng.standard_normal(size=(COMPONENTS, COMPONENTS)))[0])
    X, X_te, Y, Y_te = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    return X, Y, X_te, Y_te

def main():
    X, Y, X_te, Y_te=linear_dataset()#S*1
    _,S,_=np.linalg.svd(X.T@Y)
    print()

if __name__ == "__main__":
    main()