from cca_zoo.data import generate_covariance_data
from sklearn.model_selection import train_test_split
import numpy as np

def exponential_dataset(cca=False,random_state=0):
    N=1000
    COMPONENTS = 50
    if cca:
        (X, Y), _ = generate_covariance_data(
            1000,
            [COMPONENTS, COMPONENTS],
            latent_dims=50,
            correlation=1,
            decay=0.9,
            structure="toeplitz",
            sigma=0.5,
            random_state=random_state,
        )
    else:
        k = 10**np.linspace(0, 3, COMPONENTS + 1)
        Z = np.linalg.qr(np.random.rand(N, COMPONENTS))[0] * k[1:]
        X = Z @ np.linalg.pinv(np.linalg.qr(np.random.rand(COMPONENTS, COMPONENTS))[0])
        Y = Z @ np.linalg.pinv(np.linalg.qr(np.random.rand(COMPONENTS, COMPONENTS))[0])
    X, X_te, Y, Y_te = train_test_split(X, Y, test_size=0.2)
    return X, Y, X_te, Y_te