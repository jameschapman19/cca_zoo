from ccagame import pls
from jaxline import platform
import functools
from os import environ
from absl import app, flags
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jaxline import platform
from jaxline.base_config import get_base_config
import argparse
import jax.numpy as jnp

# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game

# These are the defaults for the above arguments
DEVICES = 4
N_COMPONENTS = 4
BATCH_SIZE = None
LEARNING_RATE = 1e-7
MODEL = "game"
# This is used to turn name of model on command line into model class
MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "oja": pls.Oja,
    "power": pls.StochasticPower,
}
TRAINING_STEPS = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=DEVICES)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--training_steps", type=int, default=TRAINING_STEPS)
    return parser.parse_args()


# THIS IS ALL OF THE PARAMETERS OF JAXLINE EXPERIMENT
# ITS BASICALLY A DICTIONARY
def get_config(
    data,
    dims=None,
    num_devices=1,
    n_components=1,
    log_tensors_interval=1,
    log_train_data_interval=1,
    training_steps=100,
    batch_size=None,
    correct_eigenvectors=None,
    **kwargs
):
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "n_components": n_components,
        "num_devices": num_devices,
        "dims": dims,
        "data": data,
        "batch_size": batch_size,
        "correct_eigenvectors": correct_eigenvectors,
        **kwargs,
    }
    config.training_steps = training_steps
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = log_tensors_interval
    config.log_train_data_interval = log_train_data_interval
    config.lock()
    return config


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    args = parse_args()
    environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices}"
    FLAGS = flags.FLAGS
    X, _, X_te, _ = mnist()
    Y=X[:, 400:]
    X=X[:, :400]
    input_data_iterator = data_stream(
        X, Y=Y, batch_size=args.batch_size
    )
    correct_U, correct_S, correct_V = jnp.linalg.svd(X.T @ Y)
    correct_U = correct_U[:, :args.n_components]
    correct_V = correct_V[:args.n_components, :].T
    FLAGS.config = get_config(
        input_data_iterator,
        dims=[400, 384],
        num_devices=args.devices,
        n_components=args.n_components,
        training_steps=args.training_steps,
        correct_eigenvectors=[correct_U,correct_V],
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, MODEL_DICT[args.model]))
