from typing import Iterable

import torch

from . import _GenerativeMixin
from .._base import BaseDeep


class DVCCA(BaseDeep, _GenerativeMixin):
    """
    A class used to fit a DVCCA model.

    References
    ----------
    Wang, Weiran, et al. 'Deep variational canonical correlation analysis.' arXiv
    preprint arXiv:1610.03454 (2016).
    https: // arxiv.org / pdf / 1610.03454.pdf
    https: // github.com / pytorch / examples / blob / master / vae / main.py

    """

    def __init__(
            self,
            *args,
            decoders=None,
            private_encoders: Iterable = None,
            latent_dropout=0,
            img_dim=None,
            recon_loss_type="mse",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_dim = img_dim
        self.latent_dropout = torch.nn.Dropout(p=latent_dropout)
        self.decoders = torch.nn.ModuleList(decoders)
        if private_encoders:
            self.private_encoders = torch.nn.ModuleList(private_encoders)
        else:
            self.private_encoders = None
        self.recon_loss_type = recon_loss_type

    def forward(self, views, mle=True, **kwargs):
        """
        Forward method for the model. Outputs latent encoding for each view

        :param views:
        :param kwargs:
        :return:
        """
        z = {}
        # Used when we get reconstructions
        z["mu_shared"], z["logvar_shared"] = self._encode(views)
        z["shared"] = self._sample(z["mu_shared"], z["logvar_shared"], mle)
        if self.private_encoders is not None:
            z["mu_private"], z["logvar_private"] = self._encode_private(views)
            z["private"] = [
                self._sample(mu_, logvar_, mle)
                for mu_, logvar_ in zip(z["mu_private"], z["logvar_private"])
            ]
        return z

    def _sample(self, mu, logvar, mle):
        """

        :param mu:
        :param logvar:
        :param mle: whether to return the maximum likelihood (i.e. mean) or whether
        to sample
        :return: a sample from latent vector
        """
        if mle:
            return mu
        else:
            return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def _encode(self, views):
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            mu_i, logvar_i = encoder(views[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return torch.stack(mu).sum(dim=0), torch.stack(logvar).sum(dim=0)

    def _encode_private(self, views):
        mu = []
        logvar = []
        for i, private_encoder in enumerate(self.private_encoders):
            mu_i, logvar_i = private_encoder(views[i])
            mu.append(mu_i)
            logvar.append(logvar_i)
        return mu, logvar

    def _decode(self, z, uncertainty=False, **kwargs):
        x = []
        if uncertainty:
            z["shared"] = z["logvar_shared"]
            if self.private_encoders is not None:
                z["private"] = z["logvar_private"]
        for i, decoder in enumerate(self.decoders):
            if "private" in z:
                x_i = decoder(
                    torch.cat(
                        (
                            self.latent_dropout(z["shared"]),
                            self.latent_dropout(z["private"][i]),
                        ),
                        dim=-1,
                    )
                )
            else:
                x_i = decoder(self.latent_dropout(z["shared"]))
            x.append(x_i)
        return x

    def minibatch_loss(self, batch, **kwargs):
        representations = self(batch["views"], mle=False)
        recons = self._decode(representations)
        loss = dict()
        loss["reconstruction"] = torch.stack(
            [
                self.recon_loss(x, recon, loss_type=self.recon_loss_type)
                for x, recon in zip(batch["views"], recons)
            ]
        ).sum()
        loss["kl shared"] = (
                self.kl_loss(representations["mu_shared"],
                             representations["logvar_shared"]) / batch["views"][
                    0].numel()
        )
        if "private" in representations:
            loss["kl private"] = torch.stack(
                [
                    self.kl_loss(mu_, logvar_) / batch["views"][0].numel()
                    for mu_, logvar_ in zip(representations["mu_private"],
                                            representations["logvar_private"])
                ]
            ).sum()
        loss["objective"] = torch.stack(tuple(loss.values())).sum()
        return loss

    @torch.no_grad()
    def transform(
            self,
            loader: torch.utils.data.DataLoader,
    ):
        self.eval()  # Ensure the model is in evaluation mode
        representations_shared = []
        representations_private = []
        for batch_idx, batch in enumerate(loader):
            views = [view.to(self.device) for view in batch["views"]]
            representations = self(views)
            representations_shared.append(representations["shared"].cpu().detach())
            if "private" in representations:
                representations_private.append(
                    [representation.cpu().detach() for representation in
                     representations["private"]])
        representations_shared = {
            "shared": torch.vstack(representations_shared).numpy()}
        if representations_private:
            representations_private = [torch.vstack(representation).numpy()
                                       for representation in representations_private
                                       ]
        return representations_shared, representations_private
