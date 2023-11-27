def torch_cross_cov(A, B):
    A = A.T
    B = B.T

    A = A - A.mean(dim=1, keepdim=True)
    B = B - B.mean(dim=1, keepdim=True)

    C = A @ B.T
    return C / (A.size(1) - 1)
