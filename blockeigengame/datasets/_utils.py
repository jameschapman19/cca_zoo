import numpy as np

def get_num_batches(X, Y=None, batch_size=None):
    num = X.shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(X, Y=None, batch_size=0, random_state=0):
    num = X.shape[0]
    if batch_size == 0:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(random_state)
    while True:
        perm = rng.permutation(num)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            if Y is None:
                yield jnp.array(X[batch_idx])
            else:
                yield np.array(X[batch_idx]), np.array(Y[batch_idx])

def get_training_steps(data, epochs, batch_size):
    if data == "mnist":
        if batch_size == 0:
            return epochs
        else:
            return 60000 * epochs // batch_size
    elif data in ["linear", "exponential"]:
        if batch_size == 0:
            return epochs
        else:
            return 1000 * epochs // batch_size
    elif data == "xrmb":
        if batch_size == 0:
            return epochs
        else:
            return 1429236 * epochs // batch_size
    else:
        raise NotImplementedError


def demean(X, X_te, Y=None, Y_te=None):
    mean = X.mean(axis=0)
    X -= mean
    X_te -= mean
    if Y is not None:
        mean = Y.mean(axis=0)
        Y -= mean
        Y_te -= mean
        return X, X_te, Y, Y_te
    return X, X_te
