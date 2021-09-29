import os
from ccagame import datasets
from ccagame.pls import Game, SGD, Incremental, Batch, Numpy, MSG
import numpy as np
import os
import wandb

# Set up your default hyperparameters

hyperparameter_defaults = dict(
    batch_size=200,
)


def main():
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    batch_size = config.batch_size
    n_components = config.n_components
    lr = config.lr
    if config.model == 'sgd':
        sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components,
                  verbose=True).fit(
            train_1,
            train_2)
        print("\n Eigenvalues calculated using sgd are :\n", sgd.score(train_1, train_2))
        print("\n Time :\n", sgd.fit_time)
        np.save(f'sgd_{batch_size}', sgd.obj)
    elif config.model == 'game':
        game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components, verbose=True,
                    mu=True).fit(train_1, train_2)
        print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
        print("\n Time :\n", game.fit_time)
        np.save(f'game_{batch_size}', game.obj)
    elif config.model == 'game':
        incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=n_components, verbose=True).fit(train_1,
                                                                                                                  train_2)
        print("\n Eigenvalues calculated using incremental are :\n", incremental.score(train_1, train_2))
        print("\n Time :\n", incremental.fit_time)
        np.save('inc', incremental.obj)
    elif config.model == 'game':
        msg = MSG(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=n_components, verbose=True).fit(
            train_1,
            train_2)
        print("\n Eigenvalues calculated using msg are :\n", msg.score(train_1, train_2))
        print("\n Time :\n", msg.fit_time)
        np.save('msg', msg.obj)
    elif config.model == 'game':
        batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=n_components, verbose=True).fit(train_1, train_2)
        print("\n Eigenvalues calculated using batch are :\n", batch.score(train_1, train_2))
        print("\n Time :\n", batch.fit_time)


if __name__ == '__main__':
    main()
