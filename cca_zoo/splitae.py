"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from cca_zoo.dcca import DCCA_base
from cca_zoo.deep_models import BaseEncoder, Encoder, BaseDecoder, Decoder


class SplitAE(DCCA_base):
    def __init__(self, latent_dims: int, encoder: BaseEncoder = Encoder,
                 decoders: Iterable[BaseDecoder] = (Decoder, Decoder), learning_rate=1e-3, lam=0.5,
                 post_transform=True, schedulers: Iterable = None, optimizers: Iterable = None):
        super().__init__(latent_dims)
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.lam = lam
        if optimizers is None:
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoders.parameters()),
                                        lr=learning_rate)
        else:
            self.optimizer = optimizers
        self.schedulers = []
        if schedulers:
            self.schedulers.extend(schedulers)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoders.parameters()),
                                    lr=learning_rate)

    def update_weights(self, *args):
        self.optimizer.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = self.encoder(args[0])
        return z

    def decode(self, z):
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(z))
        return tuple(recon)

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(z)
        recon_loss = self.recon_loss(args, recon)
        return recon_loss

    @staticmethod
    def recon_loss(x, recon):
        recons = [F.mse_loss(recon[i], x[i], reduction='sum') for i in range(len(x))]
        return torch.stack(recons).sum(dim=0)
