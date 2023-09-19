import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from cca_zoo.deep.objectives import MCCA


class BatchValidationCorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        val_corr = pl_module.score(trainer.val_dataloaders).sum()
        pl_module.log(
            "val/corr",
            val_corr,
        )


class BatchTrainCorrelationCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_corr = pl_module.score(trainer.train_dataloader).sum()
        pl_module.log(
            "train/corr",
            train_corr,
        )


class MinibatchTrainCorrelationCallback(Callback):
    mcca = MCCA()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        with torch.no_grad():
            train_corr = self.mcca.loss(pl_module(batch["views"])).sum()
            pl_module.log(
                "train/corr",
                train_corr,
            )


class MinibatchValidationCorrelationCallback(Callback):
    mcca = MCCA()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        with torch.no_grad():
            val_corr = self.mcca.loss(pl_module(batch["views"])).sum()
            pl_module.log(
                "val/corr",
                val_corr,
            )
