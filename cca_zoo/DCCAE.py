from abc import ABC

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import cca_zoo.deep_models
import cca_zoo.objectives

"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""


class DCCAE(nn.Module, ABC):

    def __init__(self, input_size_1: int, input_size_2: int, hidden_layer_sizes_1: list = None,
                 hidden_layer_sizes_2: list = None, latent_dims: int = 2,
                 lam=0, loss_type: str = 'cca', model_1: str = 'fcn', model_2: str = 'fcn', learning_rate=1e-3):
        super(DCCAE, self).__init__()

        if model_1 == 'fcn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [128]
            self.encoder_1 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
            self.decoder_1 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_1, latent_dims, input_size_1).double()
        elif model_1 == 'cnn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [1, 1, 1]
            self.encoder_1 = cca_zoo.deep_models.CNN_Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
            self.decoder_1 = cca_zoo.deep_models.CNN_Decoder(hidden_layer_sizes_1, latent_dims, input_size_1).double()
        elif model_1 == 'brainnet':
            self.encoder_1 = cca_zoo.deep_models.BrainNetCNN_Encoder(input_size_1, latent_dims).double()
            self.decoder_1 = cca_zoo.deep_models.BrainNetCNN_Decoder(latent_dims, input_size_1).double()

        if model_2 == 'fcn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [128]
            self.encoder_2 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()
            self.decoder_2 = cca_zoo.deep_models.Decoder(hidden_layer_sizes_2, latent_dims, input_size_2).double()
        elif model_2 == 'cnn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [1, 1, 1]
            self.encoder_2 = cca_zoo.deep_models.CNN_Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()
            self.decoder_2 = cca_zoo.deep_models.CNN_Decoder(hidden_layer_sizes_2, latent_dims, input_size_2).double()
        elif model_2 == 'brainnet':
            self.encoder_2 = cca_zoo.deep_models.BrainNetCNN_Encoder(input_size_2, latent_dims).double()
            self.decoder_2 = cca_zoo.deep_models.BrainNetCNN_Decoder(latent_dims, input_size_2).double()

        self.latent_dims = latent_dims

        if loss_type == 'cca':
            self.cca_objective = cca_zoo.objectives.cca(self.latent_dims)
        if loss_type == 'gcca':
            self.cca_objective = cca_zoo.objectives.gcca(self.latent_dims)
        if loss_type == 'mcca':
            self.cca_objective = cca_zoo.objectives.mcca(self.latent_dims)
        self.lam = lam
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        return z_1, z_2

    def decode(self, z_1, z_2):
        x_1_recon = self.decoder_1(z_1)
        x_2_recon = self.decoder_2(z_2)
        return x_1_recon, x_2_recon

    def forward(self, x_1, x_2):
        z = self.encode(x_1, x_2)
        recon = self.decode(*z)
        return z, recon

    def update_weights(self, x_1, x_2):
        self.optimizer.zero_grad()
        loss = self.loss(x_1, x_2)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, x_1, x_2):
        z = self.encode(x_1, x_2)
        recon = self.decode(*z)
        recon_loss = self.recon_loss(x_1, x_2, *recon)
        return self.lam * recon_loss + self.cca_objective.loss(*z)

    def recon_loss(self, x_1, x_2, x_1_recon, x_2_recon):
        recon_1 = F.mse_loss(x_1_recon, x_1, reduction='sum')
        recon_2 = F.mse_loss(x_2_recon, x_2, reduction='sum')
        return recon_1 + recon_2


class DGCCAE(nn.Module, ABC):
    def __init__(self, *args, hidden_layer_sizes=None, latent_dims=2, lam=0, loss_type='gcca', learning_rate=1e-3):
        super(DGCCAE, self).__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [[128] for arg in args]

        self.encoders = nn.ModuleList(
            [cca_zoo.deep_models.Encoder(hidden_layer_sizes[i], arg, latent_dims).double() for i, arg in
             enumerate(args)])

        self.decoders = nn.ModuleList(
            [cca_zoo.deep_models.Decoder(hidden_layer_sizes[i], latent_dims, arg).double() for i, arg in
             enumerate(args)])

        self.latent_dims = latent_dims

        if loss_type == 'gcca':
            self.cca_objective = cca_zoo.objectives.gcca(self.latent_dims)
        elif loss_type == 'mcca':
            self.cca_objective = cca_zoo.objectives.mcca(self.latent_dims)
        self.lam = lam

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, *args):
        z = []
        for i, arg in enumerate(args):
            z.append(self.encoders[i](arg))
        return tuple(z)

    def decode(self, *args):
        x = []
        for i, arg in enumerate(args):
            x.append(self.decoders[i](arg))
        return tuple(x)

    def forward(self, *args):
        z = self.encode(*args)
        recon = self.decode(*z)
        return recon

    def update_weights(self, *args):
        self.optimizer.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(*z)
        return self.lam * self.recon_loss(args, recon) + self.cca_objective.loss(*z)

    def recon_loss(self, views, recons):
        return torch.sum(torch.stack([F.mse_loss(view, recons[i], reduction='sum') for i, view in enumerate(views)]))
