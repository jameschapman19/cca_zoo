from abc import abstractmethod

import numpy as np


class _ProxUpdate:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def __call__(self, X, Y, W):
        W = self.prox(W - self.alpha * X.T @ (X @ W - Y))
        return W

    @abstractmethod
    def prox(self, W):
        raise NotImplementedError

    @abstractmethod
    def cost(self, X, W):
        raise NotImplementedError


class ProxNone:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def __call__(self, X, Y, W):
        W = self.prox(W - self.alpha * X.T @ (X @ W - Y))
        return W

    def prox(self, W):
        return W

    @abstractmethod
    def cost(self, X, W):
        return 0


class ProxLasso(_ProxUpdate):
    def __init__(self, alpha=1e-3, gamma=None):
        super().__init__(alpha)
        self.gamma = gamma

    def prox(self, W):
        if isinstance(self.gamma, (int, float)):
            self.gamma = [self.gamma] * W.shape[1]
        for k in range(W.shape[1]):
            W[:, k] = soft_threshold(W[:, k], self.gamma[k])
        return W

    def cost(self, X, W):
        return np.dot(np.linalg.norm(W, axis=0, ord=1), np.array(self.gamma))


class ProxElastic(_ProxUpdate):
    def __init__(self, alpha=1e-3, gamma=None, lam=None):
        super().__init__(alpha)
        self.gamma = gamma
        self.lam = lam

    def prox(self, W):
        if isinstance(self.gamma, (int, float)):
            self.gamma = [self.gamma] * W.shape[1]
        if isinstance(self.lam, (int, float)):
            self.lam = [self.lam] * W.shape[1]
        for k in range(W.shape[1]):
            W[:, k] = soft_threshold(W[:, k], self.gamma[k])
            W[:, k] /= 1 + self.gamma[k] * self.lam[k]
        return W

    def cost(self, X, W):
        return np.dot(np.linalg.norm(W, axis=0, ord=1), np.array(self.gamma))


class ProxPos(_ProxUpdate):
    def __init__(self, alpha=1e-3):
        super().__init__(alpha)

    def prox(self, W):
        W[W < 0] = 0
        return W

    def cost(self, X, W):
        return 0


class Prox21(_ProxUpdate):
    def __init__(self, alpha=1e-3, gamma=None):
        super().__init__(alpha)
        self.gamma = gamma

    def prox(self, W):
        for k in range(W.shape[0]):
            if np.linalg.norm(W[k, 0]) < self.gamma:
                W[k, 0] = 0
            else:
                W[k, 0] = 1 - self.gamma / np.linalg.norm(W[k, 0])
        return W

    def cost(self, X, W):
        return np.dot(np.linalg.norm(W, axis=0, ord=1), np.array(self.gamma))


class ProxFrobenius(_ProxUpdate):
    def __init__(self, alpha=1e-3, gamma=0.0):
        super().__init__(alpha)
        self.gamma = gamma

    def prox(self, W):
        W = W - self.alpha * self.gamma * np.eye(W.shape[0]) @ W
        return W

    def cost(self, X, W):
        return (self.gamma / 2) * np.linalg.norm(W)


def soft_threshold(data, value, positive=False, **kwargs):
    if positive:
        data[data < 0] = 0
    return np.sign(data) * np.maximum(np.abs(data) - value, 0)


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
