"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from cca_zoo.dcca import DCCA


class DCCAE(DCCA):

    def __init__(self, latent_dims: int, input_sizes: list, objective=None, encoder_models=None, encoder_args=None,
                 decoder_models=None, decoder_args=None,
                 learning_rate=1e-3, lam=0.5):
        super(DCCAE, self).__init__(latent_dims, input_sizes)
        self.encoders = nn.ModuleList(
            [model(input_sizes[i], latent_dims, **encoder_args[i]) for i, model in
             enumerate(encoder_models)])
        self.decoders = nn.ModuleList(
            [model(latent_dims, input_sizes[i], **decoder_args[i]) for i, model in
             enumerate(decoder_models)])
        self.lam = lam
        self.objective = objective(latent_dims)
        self.optimizer = optim.Adam(list(self.encoders.parameters()) + list(self.decoders.parameters()),
                                    lr=learning_rate)

    def update_weights(self, *args):
        """
        :param args:
        :return:
        """
        self.optimizer.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def decode(self, *args):
        """
        :param args:
        :return:
        """
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(args[i]))
        return tuple(recon)

    def loss(self, *args):
        """
        :param args:
        :return:
        """
        z = self.encode(*args)
        recon = self.decode(*z)
        recon_loss = self.recon_loss(args, recon)
        return self.lam * recon_loss + self.objective.loss(*z)

    @staticmethod
    def recon_loss(x, recon):
        """
        :param x:
        :param recon:
        :return:
        """
        recons = [F.mse_loss(recon[i], x[i], reduction='sum') for i in range(len(x))]
        return torch.stack(recons).sum(dim=0)
