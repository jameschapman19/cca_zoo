import numpy as np
from functools import partial


class l1:
    def __init__(self, tau, positive=False):
        self.tau = tau
        self.positive = positive

    def __call__(self, weight, lr):
        return np.where(
            weight > self.tau * lr,
            weight - self.tau * lr,
            np.where(weight < -self.tau * lr, weight + self.tau * lr, 0),
        )

    def cost(self, weight):
        return self.tau * np.linalg.norm(weight, 1)


class l0:
    def __init__(self, tau, positive=False):
        self.tau = tau
        self.positive = positive

    def __call__(self, weight, lr):
        return np.where(0.5 * weight**2 > self.tau * lr, weight, 0)

    def cost(self, weight):
        return self.tau * np.linalg.norm(weight, 1)


PROXIMAL_OPERATORS = {"l1": l1, "l0": l0}


def _proximal_operators(proximal, **params):
    if callable(proximal):
        return partial(proximal, **params)
    else:
        return PROXIMAL_OPERATORS[proximal](**params)


def soft_threshold(data, value, positive=False, **kwargs):
    if positive:
        data[data < 0] = 0
    return np.sign(data) * np.maximum(np.abs(data) - value, 0)


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
