from abc import abstractmethod
from typing import Iterable

import numpy as np
import torch
from torch import nn


class _DCCA_base:
    def __init__(self, latent_dims: int, optimizer: torch.optim.Optimizer, scheduler=None, clip_value=float('inf')):
        self.latent_dims = latent_dims
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_value = clip_value

    def update_weights(self, *args):
        """
        A complete update of the weights used every batch
        :param args: batches for each view separated by commas
        :return:
        """
        if type(self.optimizer) == torch.optim.LBFGS:
            def closure():
                """
                Required by LBFGS optimizer
                """
                self.optimizer.zero_grad()
                loss = self.loss(*args)
                loss.backward()
                return loss

            nn.utils.clip_grad_value_(self.parameters(), clip_value=self.clip_value)
            self.optimizer.step(closure)
            loss = closure()
        else:
            self.optimizer.zero_grad()
            loss = self.loss(*args)
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), clip_value=self.clip_value)
            self.optimizer.step()
        return loss

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        pass

    def post_transform(self, *z_list, train=False) -> Iterable[np.ndarray]:
        return z_list
