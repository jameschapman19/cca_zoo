import torch
from torch import nn
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class PLSGame(LightningModule):
    def __init__(self, n_components, x_features, y_features, learning_rate=1):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.n_components = n_components
        self.x_features = x_features
        self.y_features = y_features
        self.x_players = nn.ParameterList([nn.Parameter(torch.ones(x_features, 1)) for _ in range(n_components)])
        self.y_players = nn.ParameterList([nn.Parameter(torch.ones(y_features, 1)) for _ in range(n_components)])
        self.normalize_players()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        x_optimizers = [torch.optim.SGD(nn.ParameterList([parameters]), lr=lr) for parameters in self.x_players]
        y_optimizers = [torch.optim.SGD(nn.ParameterList([parameters]), lr=lr) for parameters in self.y_players]
        return x_optimizers + y_optimizers

    def deflation(self, k, view='x'):
        if view == 'x':
            if k == 0:
                return torch.eye(self.x_features)
            return torch.eye(self.x_features) - torch.sum(
                torch.stack([player.detach() @ player.detach().T for player in self.x_players[:k]]), dim=0)
        if view == 'y':
            if k == 0:
                return torch.eye(self.y_features)
            return torch.eye(self.y_features) - torch.sum(
                torch.stack([player.detach() @ player.detach().T for player in self.y_players[:k]]), dim=0)

    def forward(self, x, y):
        x_obj = []
        y_obj = []
        for k in range(self.n_components):
            Cxy = x.T @ y
            x_obj.append(-self.x_players[k].T @ self.deflation(k, 'x') @ Cxy @ self.y_players[k].detach())
            y_obj.append(-self.y_players[k].T @ self.deflation(k, 'y') @ Cxy.T @ self.x_players[k].detach())
        return x_obj, y_obj

    def tv(self, x, y):
        U = x @ torch.hstack([parameter for parameter in self.x_players])
        V = y @ torch.hstack([parameter for parameter in self.y_players])
        eigvals = torch.linalg.eigvals(U.T @ V)
        return eigvals

    def normalize_players(self):
        with torch.no_grad():
            for parameter in self.x_players:
                parameter.div_(torch.norm(parameter))
            for parameter in self.y_players:
                parameter.div_(torch.norm(parameter))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_objs, y_objs = self(x, y)
        opts = self.optimizers(use_pl_optimizer=False)
        x_opts, y_opts = opts[:self.n_components], opts[self.n_components:]
        [self.manual_backward(x_obj) for x_obj in x_objs]
        [self.manual_backward(y_obj) for y_obj in y_objs]
        [x_opt.step() for x_opt in x_opts]
        [y_opt.step() for y_opt in y_opts]
        [x_opt.zero_grad() for x_opt in x_opts]
        [y_opt.zero_grad() for y_opt in y_opts]
        self.normalize_players()
        loss = self.tv(x, y)
        [self.log(f'train_loss_{component}', loss[component]) for component in range(self.n_components)]

    def val_step(self, batch, batch_idx):
        x, y = batch
        loss = self.tv(x, y)
        [self.log(f'val_loss_{component}', loss[component]) for component in range(self.n_components)]


def main():
    x_features = 10
    y_features = 11
    n = 10000
    batch_size=100

    x_train, y_train = np.random.rand(2 * n, x_features), np.random.rand(2 * n, y_features)

    train = TensorDataset(torch.tensor(x_train[:n]).float(), torch.tensor(y_train[:n]).float())
    train = DataLoader(train, batch_size=n)

    val = TensorDataset(torch.tensor(x_train[n:]).float(), torch.tensor(y_train[n:]).float())
    val = DataLoader(val, batch_size=n)
    # init model
    ccagame = CCAGame(3, x_features, y_features, learning_rate=1)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20, log_every_n_steps=1)

    # Train the model ⚡
    trainer.fit(ccagame, train, val)

    print("exact",np.linalg.svd(x_train[:n].T@y_train[:n])[1])

    print()


if __name__ == '__main__':
    main()
