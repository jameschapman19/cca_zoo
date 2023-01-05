from abc import abstractmethod

import numpy as np

def soft_threshold(data, value, positive=False, **kwargs):
    if positive:
        data[data < 0] = 0
    return np.sign(data) * np.maximum(np.abs(data) - value, 0)


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
