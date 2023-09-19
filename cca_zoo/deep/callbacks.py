from pytorch_lightning import Callback, LightningModule, Trainer


class ValidationCorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        val_corr = pl_module.score(trainer.val_dataloaders).sum()
        pl_module.log(
            "val/corr",
            val_corr,
        )


class TrainCorrelationCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_corr = pl_module.score(trainer.train_dataloader).sum()
        pl_module.log(
            "train/corr",
            train_corr,
        )
