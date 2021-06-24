from abc import abstractmethod

import torch


class _DCCA_base(torch.nn.Module):
    def __init__(self, latent_dims: int, optimizer: torch.optim.Optimizer, scheduler=None):
        super(_DCCA_base, self).__init__()
        self.latent_dims = latent_dims
        self.optimizer = optimizer
        self.schedulers = scheduler

    def update_weights(self, *args):
        """
        A complete update of the weights used every batch
        :param args: batches for each view separated by commas
        :return:
        """
        if type(self.optimizer) == torch.optim.LBFGS:
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss(*args)
                loss.backward()
                return loss

            self.optimizer.step(closure)
            loss = closure()
        else:
            self.optimizer.zero_grad()
            loss = self.loss(*args)
            loss.backward()
            self.optimizer.step()
        return loss

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        pass

    def post_transform(self, *z_list, train=False):
        return z_list
