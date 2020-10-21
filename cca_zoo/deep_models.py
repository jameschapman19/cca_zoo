from abc import ABC
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
I've included some standard models for the DCCA encoders and DCCAE decoders. 
We have a FCN, CNN and GNN which we can compare.
Particular thanks for the basic structure to:
https://github.com/Michaelvll/DeepCCA/blob/master/DeepCCAModels.py
'''


class Encoder(nn.Module, ABC):
    def __init__(self, layer_sizes, input_size: int, output_size: int):
        super(Encoder, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes + [output_size]
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id]),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module, ABC):
    def __init__(self, layer_sizes, input_size: int, output_size: int):
        super(Decoder, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes + [output_size]
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id]),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNN_Encoder(nn.Module, ABC):
    def __init__(self, layer_sizes, input_size: int, output_size: int, kernel_sizes=None, stride=None, padding=None):
        super(CNN_Encoder, self).__init__()
        # assume square input
        layers = []
        layer_sizes = layer_sizes + [output_size]

        if kernel_sizes is None:
            kernel_sizes = [5] * (len(layer_sizes) - 1)
        if stride is None:
            stride = [1] * (len(layer_sizes) - 1)
        if padding is None:
            padding = [2] * (len(layer_sizes) - 1)

        current_size = input_size
        current_channels = 1
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.Linear(int(current_size * current_size * current_channels), layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(  # input shape (1, current_size, current_size)
                    nn.Conv2d(
                        in_channels=current_channels,  # input height
                        out_channels=layer_sizes[l_id],  # n_filters
                        kernel_size=kernel_sizes[l_id],  # filter size
                        stride=stride[l_id],  # filter movement/step
                        padding=padding[l_id],
                        # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                        # stride=1
                    ),  # output shape (out_channels, current_size, current_size)
                    nn.ReLU(),  # activation
                    nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
                ))
                current_size = current_size / 2
                current_channels = layer_sizes[l_id]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = x.reshape((x.shape[0], -1))
                x = layer(x)
            else:
                x = layer(x)
        return x


class CNN_Decoder(nn.Module, ABC):
    def __init__(self, layer_sizes, input_size: int, output_size: int, kernel_sizes=None, stride=None, padding=None):
        super(CNN_Decoder, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes + [output_size]

        if kernel_sizes is None:
            kernel_sizes = [4] * (len(layer_sizes) - 1)
        if stride is None:
            stride = [2] * (len(layer_sizes) - 1)
        if padding is None:
            padding = [1] * (len(layer_sizes) - 1)

        current_size = output_size
        current_channels = 1
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.Linear(input_size, int(current_size * current_size * current_channels)),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.ReLU(),  # input shape (1, current_size, current_size)
                    nn.ConvTranspose2d(
                        in_channels=layer_sizes[l_id],  # input height
                        out_channels=current_channels,
                        kernel_size=kernel_sizes[l_id],
                        stride=stride[l_id],  # filter movement/step
                        padding=padding[l_id],
                        # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                        # stride=1
                    )))
                current_size = current_size / 2
                current_channels = layer_sizes[l_id]
        self.layers = nn.ModuleList(layers[::-1])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                x = x.reshape((x.shape[0], self.layers[i + 1][1].in_channels, -1))
                x = x.reshape(
                    (x.shape[0], self.layers[i + 1][1].in_channels, int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1]))))
            else:
                x = layer(x)
        return x


# https://github.com/nicofarr/brainnetcnnVis_pytorch/blob/master/BrainNetCnnGoldMSI.py
class E2EBlock(nn.Module, ABC):
    def __init__(self, in_planes, planes, size, bias=False):
        super(E2EBlock, self).__init__()

        self.d = size
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class E2EBlock_reverse(nn.Module, ABC):
    def __init__(self, in_planes, planes, size, bias=False):
        super(E2EBlock_reverse, self).__init__()

        self.d = size
        self.cnn1 = torch.nn.ConvTranspose2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.ConvTranspose2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


# BrainNetCNN Network for fitting Gold-MSI on LSD dataset
class BrainNetCNN_Encoder(nn.Module, ABC):
    def __init__(self, input_size: int, output_size: int):
        super(BrainNetCNN_Encoder, self).__init__()
        self.d = input_size
        self.e2econv1 = E2EBlock(1, 32, self.d, bias=True)
        self.e2econv2 = E2EBlock(32, 64, self.d, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, output_size)

    def forward(self, x):  # 16,1,200,200
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)  # 16,32,200,200
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)  # 16,64,200,200
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)  # 16,1,200,1
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)  # 16,256,1,1
        out = out.view(out.size(0), -1)  # 16,256
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        return out


class BrainNetCNN_Decoder(nn.Module, ABC):
    def __init__(self, input_size: int, output_size: int):
        super(BrainNetCNN_Decoder, self).__init__()
        self.d = output_size
        self.e2econv1 = E2EBlock(32, 1, self.d, bias=True)
        self.e2econv2 = E2EBlock(64, 32, self.d, bias=True)
        self.E2N = torch.nn.ConvTranspose2d(1, 64, (1, self.d))
        self.N2G = torch.nn.ConvTranspose2d(256, 1, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(30, 128)
        self.dense3 = torch.nn.Linear(input_size, 30)

    def forward(self, x):
        out = F.dropout(F.leaky_relu(self.dense3(x), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv1(out), negative_slope=0.33)
        return out
