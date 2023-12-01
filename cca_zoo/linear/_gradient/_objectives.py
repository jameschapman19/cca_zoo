from typing import List
import numpy as np

from cca_zoo._utils._cross_correlation import cross_cov


def CCA_AB(representations: List[np.ndarray]):
    latent_dimensions = representations[0].shape[1]
    A = np.zeros(
        (latent_dimensions, latent_dimensions)
    )  # initialize the cross-covariance matrix
    B = np.zeros(
        (latent_dimensions, latent_dimensions)
    )  # initialize the auto-covariance matrix
    for i, zi in enumerate(representations):
        B += np.cov(zi.T)  # In-place addition
        for j, zj in enumerate(representations):
            A += cross_cov(zi, zj, rowvar=False)  # In-place addition

    A /= len(representations)  # In-place division
    B /= len(representations)  # In-place division
    return A, B


def PLS_AB(representations: List[np.ndarray], weights: List[np.ndarray]):
    latent_dimensions = representations[0].shape[1]
    A = np.zeros(
        (latent_dimensions, latent_dimensions)
    )  # initialize the cross-covariance matrix
    B = np.zeros(
        (latent_dimensions, latent_dimensions)
    )  # initialize the auto-covariance matrix

    for i, zi in enumerate(representations):
        B += weights[i].T @ weights[i]  # In-place addition
        for j, zj in enumerate(representations):
            if i != j:
                A += cross_cov(zi, zj, rowvar=False)  # In-place addition

    A /= len(representations)  # In-place division
    B /= len(representations)  # In-place division
    return A, B
