import torch.utils.data
from torch import nn
from torch.nn import functional as F

from CCA_methods.objectives import *


class DCCA_adversary(nn.Module):
    def __init__(self, outdim_size=2, confounds=None, alpha=1):
        super(DCCA_adversary, self).__init__()
        self.confounds = confounds
        self.outdim_size = outdim_size
        # Confound discriminator
        self.discriminator_1 = nn.Linear(outdim_size * 2, self.confounds)
        #self.discriminator_2 = nn.Linear(400, self.confounds)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # controls confound discriminator weight
        self.alpha = alpha

    def forward(self, z_x, z_y):
        h = self.discriminator_1(torch.cat((z_x, z_y), dim=-1))
        return h

    def loss(self, true_confounds, confounds):
        discriminator = F.mse_loss(confounds, true_confounds) / (
                confounds.shape[0] * confounds.shape[1])
        return self.alpha * discriminator

    def breakdown(self, true_confounds, confounds):
        breakdown = F.mse_loss(confounds, true_confounds, reduction='none')
        return breakdown