from abc import abstractmethod

import torch
from torch.nn import functional as F


class _GenerativeMixin:
    def recon_loss(self, x, recon, loss="mse", reduction="mean", **kwargs):
        if loss == "mse":
            return self.mse_loss(x, recon, reduction=reduction)
        elif loss == "bce":
            return self.mse_loss(x, recon, reduction=reduction)
        elif loss == "nll":
            return self.mse_loss(x, recon, reduction=reduction)

    def mse_loss(self, x, recon, reduction="mean"):
        return F.mse_loss(recon, x, reduction=reduction)

    def bce_loss(self, x, recon, reduction="mean"):
        return F.binary_cross_entropy(recon, x, reduction=reduction)

    def nll_loss(self, x, recon, reduction="mean"):
        return F.nll_loss(recon, x, reduction=reduction)

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @abstractmethod
    def _decode(self, z, **kwargs):
        raise NotImplementedError

    def recon(self, loader: torch.utils.data.DataLoader, **kwargs):
        with torch.no_grad():
            x = []
            for batch_idx, batch in enumerate(loader):
                views = [view.to(self.device) for view in batch["views"]]
                x_ = self.detach_all(self._decode(self(views, **kwargs), **kwargs))
                x.append(x_)
        x = [torch.vstack(i).cpu().numpy() for i in zip(*x)]
        return x
