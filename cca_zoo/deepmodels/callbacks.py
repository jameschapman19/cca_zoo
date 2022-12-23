import torch
import torchvision
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule, Trainer
from torch.autograd import Variable


class CorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        pl_module.log(
            "val/corr",
            pl_module.score(trainer.val_dataloaders[0]).sum(),
        )


class GenerativeCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if hasattr(pl_module, "img_dim") and pl_module.img_dim is not None:
            z = dict()
            z["shared"] = Variable(torch.randn(64, pl_module.latent_dims))
            if pl_module.private_encoders:
                z["private"] = [Variable(torch.randn(64, pl_module.latent_dims))] * len(
                    pl_module.private_encoders
                )
            sample = pl_module._decode(z)
            sample[0] = torch.reshape(sample[0], (64,) + pl_module.img_dim)
            sample[1] = torch.reshape(sample[1], (64,) + pl_module.img_dim)
            grid1 = torchvision.utils.make_grid(sample[0])
            grid2 = torchvision.utils.make_grid(sample[1])
            pl_module.logger.experiment.add_image(
                "generated_images_1", grid1, pl_module.current_epoch
            )
            pl_module.logger.experiment.add_image(
                "generated_images_2", grid2, pl_module.current_epoch
            )
