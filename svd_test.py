import numpy as np

X = np.random.rand(100, 2)

X[:, 1] = X[:, 0]

U, S, V = np.linalg.svd(X)

print()
