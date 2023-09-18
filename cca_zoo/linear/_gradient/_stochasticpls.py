import torch

from cca_zoo.linear._gradient._ey import PLS_EY
from cca_zoo.linear._pls import PLSMixin


class PLSStochasticPower(PLS_EY, PLSMixin):
    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def training_step(self, batch, batch_idx):
        if batch is None:
            batch = dict(("views", self.data))
        for weight in self.torch_weights:
            weight.data = self._orth(weight)
        scores = self(batch["views"])
        # find the pairwise covariance between the scores
        cov = torch.cov(torch.hstack(scores).T)
        loss = torch.trace(cov[: scores[0].shape[1], scores[0].shape[1] :])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_end(self) -> None:
        for weight in self.torch_weights:
            weight.data = self._orth(weight)

    @staticmethod
    def _orth(U):
        Qu, Ru = torch.linalg.qr(U)
        Su = torch.sign(torch.sign(torch.diag(Ru)) + 0.5)
        return Qu @ torch.diag(Su)
