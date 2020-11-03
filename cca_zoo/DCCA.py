from abc import ABC

from torch import nn
from torch import optim

import cca_zoo.deep_models
import cca_zoo.objectives

"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""


class DCCA(nn.Module, ABC):

    def __init__(self, input_size_1: int, input_size_2: int, hidden_layer_sizes_1: list = None,
                 hidden_layer_sizes_2: list = None, latent_dims: int = 2,
                 lam=0, loss_type: str = 'cca', model_1: str = 'fcn', model_2: str = 'fcn', learning_rate=1e-3):
        super(DCCA, self).__init__()

        if model_1 == 'fcn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [128]
            self.encoder_1 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
        elif model_1 == 'cnn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [1, 1, 1]
            self.encoder_1 = cca_zoo.deep_models.CNN_Encoder(hidden_layer_sizes_1, input_size_1, latent_dims).double()
        elif model_1 == 'brainnet':
            self.encoder_1 = cca_zoo.deep_models.BrainNetCNN_Encoder(input_size_1, latent_dims).double()

        if model_2 == 'fcn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [128]
            self.encoder_2 = cca_zoo.deep_models.Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()
        elif model_2 == 'cnn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [1, 1, 1]
            self.encoder_2 = cca_zoo.deep_models.CNN_Encoder(hidden_layer_sizes_2, input_size_2, latent_dims).double()
        elif model_2 == 'brainnet':
            self.encoder_2 = cca_zoo.deep_models.BrainNetCNN_Encoder(input_size_2, latent_dims).double()

        self.latent_dims = latent_dims

        if loss_type == 'cca':
            self.cca_objective = cca_zoo.objectives.cca(self.latent_dims)
        elif loss_type == 'gcca':
            self.cca_objective = cca_zoo.objectives.gcca(self.latent_dims)
        elif loss_type == 'mcca':
            self.cca_objective = cca_zoo.objectives.mcca(self.latent_dims)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        return z_1, z_2

    def forward(self, x_1, x_2):
        z_1, z_2 = self.encode(x_1, x_2)
        return z_1, z_2

    def update_weights(self, x_1, x_2):
        self.optimizer.zero_grad()
        loss = self.loss(x_1, x_2)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, x_1, x_2):
        z_1, z_2 = self(x_1, x_2)
        return self.cca_objective.loss(z_1, z_2)


class DGCCA(nn.Module, ABC):
    def __init__(self, *args, hidden_layer_sizes=None, latent_dims=2, lam=0, loss_type='gcca', learning_rate=1e-3):
        super(DGCCA, self).__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [[128] for _ in args]

        self.encoders = nn.ModuleList(
            [cca_zoo.deep_models.Encoder(hidden_layer_sizes[i], arg, latent_dims).double() for i, arg in
             enumerate(args)])

        self.latent_dims = latent_dims

        if loss_type == 'gcca':
            self.cca_objective = cca_zoo.objectives.gcca(self.latent_dims)
        if loss_type == 'mcca':
            self.cca_objective = cca_zoo.objectives.mcca(self.latent_dims)
        self.lam = lam
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def encode(self, *args):
        z = []
        for i, arg in enumerate(args):
            z.append(self.encoders[i](arg))
        return tuple(z)

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def update_weights(self, x_1, x_2):
        self.optimizer.zero_grad()
        loss = self.loss(*self(x_1, x_2))
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, *args):
        z = self(args)
        return self.cca_objective.loss(*z)
