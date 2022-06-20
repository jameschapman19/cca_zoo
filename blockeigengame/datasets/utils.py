def get_training_steps(data, epochs, batch_size):
    if data == "mnist":
        if batch_size==0:
            return epochs
        else:
            return 60000 * epochs // batch_size
    elif data in ["linear", "exponential"]:
        if batch_size==0:
            return epochs
        else:
            return 1000 * epochs // batch_size
    elif data == "xrmb":
        if batch_size==0:
            return epochs
        else:
            return 1429236 * epochs // batch_size
    else:
        raise NotImplementedError

def demean(X,X_te,Y=None,Y_te=None):
    mean=X.mean(axis=0)
    X-=mean
    X_te-=mean
    if Y is not None:
        mean=Y.mean(axis=0)
        Y-=mean
        Y_te-=mean
        return X, X_te, Y, Y_te
    return X,X_te