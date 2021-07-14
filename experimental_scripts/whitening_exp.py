import numpy as np
from scipy.linalg import sqrtm

X = np.random.rand(100, 40)

W = sqrtm(np.linalg.inv(X.T @ X))
M = (X @ W).T @ (X @ W)

W_c = np.linalg.cholesky(np.linalg.inv(X.T @ X))
M_c = (X @ W_c).T @ (X @ W_c)

print()
