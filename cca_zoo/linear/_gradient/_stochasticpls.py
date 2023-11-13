import torch

from cca_zoo.deep.objectives import _PLS_PowerLoss
from cca_zoo.linear._gradient._ey import PLS_EY
from cca_zoo.linear._pls import PLSMixin


class PLSStochasticPower(PLS_EY, PLSMixin):
    automatic_optimization = False
    objective = _PLS_PowerLoss()

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        representations = self(batch["views"])
        loss = self.objective.loss(representations)
        for k, v in loss.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        manual_grads = self.objective.derivative(batch["views"], representations)
        for i, weights in enumerate(self.torch_weights):
            weights.grad = manual_grads[i]
        opt.step()
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        representations = self(batch["views"])
        loss = self.objective.loss(representations)
        # Logging the loss components
        for k, v in loss.items():
            self.log(
                f"val/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def on_train_batch_start(self, batch, batch_idx):
        for weight in self.torch_weights:
            weight.data = self._orth(weight)

    @staticmethod
    def _orth(U):
        Qu, Ru = torch.linalg.qr(U)
        Su = torch.sign(torch.sign(torch.diag(Ru)) + 0.5)
        return Qu @ torch.diag(Su)
