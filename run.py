import random

import numpy as np

import datasets
from ccagame.pls import Game, SGD, Incremental, Batch, MSG
import wandb

hyperparameter_defaults = dict(
    batch_size=100,
    n_components=1,
    lr=1e-3,
    epochs=1,
    dataset='mnist',
    model='sgd',
    scale=False,
    wandb=True,
)


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seeds(42)
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    if config.dataset == 'mnist':
        train, train_labels, test, test_labels = datasets.mnist()
        train_1 = train[:, :392]
        train_2 = train[:, 392:]
    elif config.dataset == 'xrmb':
        train_1, train_2, test_1, test_2 = datasets.xrmb()
    elif config.dataset == 'ukbb':
        raise NotImplementedError
    else:
        raise ValueError(f'No dataset {config.dataset}')

    batch_size = config.batch_size
    n_components = config.n_components
    lr = config.lr
    epochs = config.epochs
    scale = config.scale

    if config.model == 'sgd':
        sgd = SGD(lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components,
                  verbose=True, scale=scale, wandb=True).fit(
            train_1,
            train_2)
        print("\n Eigenvalues calculated using sgd are :\n", sgd.score(train_1, train_2))
        print("\n Time :\n", sgd.fit_time)
        np.save(f'sgd_{batch_size}', sgd.obj)
    elif config.model == 'mugame':
        game = Game(lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components, verbose=True,
                    mu=True, scale=scale, wandb=True).fit(train_1, train_2)
        print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
        print("\n Time :\n", game.fit_time)
        np.save(f'game_{batch_size}', game.obj)
    elif config.model == 'alphagame':
        game = Game(lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components, verbose=True,
                    mu=False, scale=scale, wandb=True).fit(train_1, train_2)
        print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
        print("\n Time :\n", game.fit_time)
        np.save(f'game_{batch_size}', game.obj)
    elif config.model == 'incremental':
        incremental = Incremental(lr=lr, epochs=epochs, n_components=n_components, verbose=True, scale=scale,
                                  wandb=True).fit(
            train_1,
            train_2)
        print("\n Eigenvalues calculated using incremental are :\n", incremental.score(train_1, train_2))
        print("\n Time :\n", incremental.fit_time)
        np.save('inc', incremental.obj)
    elif config.model == 'msg':
        msg = MSG(lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components,
                  verbose=True, scale=scale, wandb=True).fit(
            train_1,
            train_2)
        print("\n Eigenvalues calculated using msg are :\n", msg.score(train_1, train_2))
        print("\n Time :\n", msg.fit_time)
        np.save('msg', msg.obj)
    elif config.model == 'batch':
        batch = Batch(lr=lr, epochs=epochs, n_components=n_components, verbose=True, scale=scale, wandb=True).fit(
            train_1, train_2)
        print("\n Eigenvalues calculated using batch are :\n", batch.score(train_1, train_2))
        print("\n Time :\n", batch.fit_time)


if __name__ == '__main__':
    main()
