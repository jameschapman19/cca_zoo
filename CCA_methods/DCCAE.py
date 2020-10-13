import torch
from torch import nn
from torch.nn import functional as F

from CCA_methods.deep_models import Encoder, Decoder, CNN_Encoder, CNN_Decoder, BrainNetCNN_Encoder, \
    BrainNetCNN_Decoder
from CCA_methods.objectives import cca, mcca, gcca

"""
All of my deep architectures have forward methods inherited from pytorch as well as a method:

loss(): which calculates the loss given some inputs and model outputs i.e.

loss(inputs,model(inputs))

This allows me to wrap them all up in the deep wrapper. Obviously this isn't required but it is helpful
for standardising the pipeline for comparison
"""

class DCCAE(nn.Module):

    def __init__(self, input_size_1, input_size_2, hidden_layer_sizes_1=None, hidden_layer_sizes_2=None, outdim_size=2,
                 lam=0, loss_type='cca', model_1='fcn', model_2='fcn'):
        super(DCCAE, self).__init__()

        if model_1 == 'fcn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [128]
            self.encoder_1 = Encoder(hidden_layer_sizes_1, input_size_1, outdim_size).double()
            self.decoder_1 = Decoder(hidden_layer_sizes_1, outdim_size, input_size_1).double()
        elif model_1 == 'cnn':
            if hidden_layer_sizes_1 is None:
                hidden_layer_sizes_1 = [1, 1, 1]
            self.encoder_1 = CNN_Encoder(hidden_layer_sizes_1, input_size_1, outdim_size).double()
            self.decoder_1 = CNN_Decoder(hidden_layer_sizes_1, outdim_size, input_size_1).double()
        elif model_1 == 'brainnet':
            self.encoder_1 = BrainNetCNN_Encoder(input_size_1, outdim_size).double()
            self.decoder_1 = BrainNetCNN_Decoder(outdim_size, input_size_1).double()

        if model_2 == 'fcn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [128]
            self.encoder_2 = Encoder(hidden_layer_sizes_2, input_size_2, outdim_size).double()
            self.decoder_2 = Decoder(hidden_layer_sizes_2, outdim_size, input_size_2).double()
        if model_2 == 'cnn':
            if hidden_layer_sizes_2 is None:
                hidden_layer_sizes_2 = [1, 1, 1]
            self.encoder_2 = CNN_Encoder(hidden_layer_sizes_2, input_size_2, outdim_size).double()
            self.decoder_2 = CNN_Decoder(hidden_layer_sizes_2, outdim_size, input_size_2).double()
        if model_2 == 'brainnet':
            self.encoder_2 = BrainNetCNN_Encoder(input_size_2, outdim_size).double()
            self.decoder_2 = BrainNetCNN_Decoder(outdim_size, input_size_2).double()

        self.outdim_size = outdim_size

        if loss_type == 'cca':
            self.cca_objective = cca(self.outdim_size)
        if loss_type == 'gcca':
            self.cca_objective = gcca(self.outdim_size)
        if loss_type == 'mcca':
            self.cca_objective = mcca(self.outdim_size)
        self.lam = lam

    def encode(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        return z_1, z_2

    def decode(self, z_1, z_2):
        x_1_recon = self.decoder_1(z_1)
        x_2_recon = self.decoder_2(z_2)
        return x_1_recon, x_2_recon

    def forward(self, x_1, x_2):
        z_1, z_2 = self.encode(x_1, x_2)
        x_1_recon, x_2_recon = self.decode(z_1, z_2)
        return z_1, z_2, x_1_recon, x_2_recon

    def loss(self, x_1, x_2, z_1, z_2, x_1_recon, x_2_recon):
        recon_1 = F.mse_loss(x_1_recon, x_1, reduction='sum')
        recon_2 = F.mse_loss(x_2_recon, x_2, reduction='sum')
        return self.lam * recon_1 + self.lam * recon_2 + self.cca_objective.loss(z_1, z_2)


class DGCCAE(nn.Module):
    def __init__(self, *args, hidden_layer_sizes=None, outdim_size=2, lam=0, loss_type='cca'):
        super(DGCCAE, self).__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [[128] for arg in args]

        self.encoders = nn.ModuleList(
            [Encoder(hidden_layer_sizes[i], arg, outdim_size).double() for i, arg in enumerate(args)])

        self.decoders = nn.ModuleList(
            [Decoder(hidden_layer_sizes[i], outdim_size, arg).double() for i, arg in enumerate(args)])

        self.outdim_size = outdim_size

        if loss_type == 'cca':
            assert len(args) == 2
            self.cca_objective = cca(self.outdim_size)
        if loss_type == 'gcca':
            self.cca_objective = gcca(self.outdim_size)
        if loss_type == 'mcca':
            self.cca_objective = mcca(self.outdim_size)
        self.lam = lam

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
        x_recon = self.decode(*z)
        return z + x_recon

    def loss(self, *args):
        return self.cca_objective.loss(*args)

    def recon_loss(self, *args):
        return self.lam * torch.sum(torch.stack([F.mse_loss(arg, arg, reduction='sum') for arg in args]))
