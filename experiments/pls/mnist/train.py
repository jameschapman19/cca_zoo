from re import I
import os
os.chdir('/mnt/c/users/chapm/PycharmProjects/ccagame')
from ccagame import pls
import functools
from os import environ
from absl import app, flags
from ccagame.utils import data_stream, log_dir
from datasets.mnist import mnist
from jaxline_fork import platform
import jax.numpy as jnp
import os
from experiments import parse_args, get_config
import wandb

# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "oja": pls.Oja,
    "power": pls.StochasticPower,
    "incremental":pls.Incremental
}


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    args = parse_args()
    wandb.init(config=args,sync_tensorboard=True)
    config = wandb.config
    environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={config.devices}"
    FLAGS = flags.FLAGS
    X, _, X_te, _ = mnist()
    Y=X[:,392:]
    X=X[:,:392]
    Y_te=X_te[:,392:]
    X_te=X_te[:,:392]
    input_data_iterator = data_stream(
        X, Y=Y, batch_size=config.batch_size
    )
    correct_U, correct_S, correct_V = jnp.linalg.svd(X.T @ Y)
    correct_U = correct_U[:, :config.n_components]
    correct_V = correct_V[:config.n_components, :].T
    FLAGS.config = get_config(
        input_data_iterator,
        dims=[X.shape[1], Y.shape[1]],
        num_devices=config.devices,
        n_components=config.n_components,
        training_steps=config.training_steps,
        correct_eigenvectors=[correct_U,correct_V],
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        model=config.model,
        holdout=[X_te,Y_te]
    )
    flags.mark_flag_as_required("config")
    #magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    os.chdir(log_dir())
    app.run(functools.partial(platform.main, MODEL_DICT[config.model]))