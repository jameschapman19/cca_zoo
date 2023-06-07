from pytorch_lightning import Callback, LightningModule, Trainer


class CorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        pl_module.log(
            "val/corr",
            pl_module.score(trainer.val_dataloaders[0]).sum(),
        )
