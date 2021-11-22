from ccagame import pls
from jaxline import platform
import functools
from os import environ
from absl import app, flags
from ccagame.utils import data_stream, log_dir
from datasets.xrmb import xrmb
from jaxline import platform
import jax.numpy as jnp
import os
from experiments import parse_args, get_config

# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
DEVICES = 4
N_COMPONENTS = 16
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
    environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices}"
    FLAGS = flags.FLAGS
    X, Y, X_te, Y_te = xrmb()
    input_data_iterator = data_stream(
        X, Y=Y, batch_size=args.batch_size
    )
    correct_U, correct_S, correct_V = jnp.linalg.svd(X.T @ Y)
    correct_U = correct_U[:, :args.n_components]
    correct_V = correct_V[:args.n_components, :].T
    FLAGS.config = get_config(
        input_data_iterator,
        dims=[X.shape[1], Y.shape[1]],
        num_devices=args.devices,
        n_components=args.n_components,
        training_steps=args.training_steps,
        correct_eigenvectors=[correct_U,correct_V],
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model=args.model,
    )
    flags.mark_flag_as_required("config")
    #magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    os.chdir(log_dir())
    app.run(functools.partial(platform.main, MODEL_DICT['game']))