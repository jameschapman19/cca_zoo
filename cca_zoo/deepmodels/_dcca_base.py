from abc import abstractmethod

import torch


class _DCCA_base(torch.nn.Module):
    def __init__(self, latent_dims: int):
        super(_DCCA_base, self).__init__()
        self.latent_dims = latent_dims
        self.schedulers = [None]

    @abstractmethod
    def update_weights(self, *args):
        """
        A complete update of the weights used every batch
        :param args: batches for each view separated by commas
        :return:
        """
        pass

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        pass

    def post_transform(self, *z_list, train=False):
        return z_list
