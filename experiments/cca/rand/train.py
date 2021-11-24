from ccagame import cca
import functools
from os import environ
from absl import app, flags
from ccagame.utils import data_stream, log_dir
from jaxline_fork import platform
import os
from experiments import parse_args, get_config
from cca_zoo.models import rCCA
import numpy as np
import jax.numpy as jnp
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": cca.Game,
}

# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    args = parse_args()
    environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices}"
    FLAGS = flags.FLAGS
    X = np.random.normal(size=(100, 11))
    X -= X.mean(axis=0)
    X/=X.std(axis=0)
    Y = np.random.normal(size=(100, 10))
    Y -= Y.mean(axis=0)
    Y/=Y.std(axis=0)
    input_data_iterator = data_stream(X, Y=Y, batch_size=args.batch_size)
    cca = rCCA(latent_dims=args.n_components, scale=False, c=0).fit(
        (np.array(X), np.array(Y))
    )
    print(
        f"TV : {cca.score((X,Y))}"
    )  # jnp.corrcoef((X@correct_U).T,(Y@correct_V).T)[args.n_components:,:args.n_components]
    correct_U = cca.weights[0]
    correct_V = cca.weights[1]
    FLAGS.config = get_config(
        input_data_iterator,
        dims=[X.shape[1], Y.shape[1]],
        num_devices=args.devices,
        n_components=args.n_components,
        training_steps=args.training_steps,
        correct_eigenvectors=[correct_U, correct_V],
        learning_rate=args.learning_rate,
        model=args.model,
        batch_size=args.batch_size,
        holdout=[X, Y],
    )
    flags.mark_flag_as_required("config")
    # magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    os.chdir(log_dir())
    # TODO THIS IS CURRENTLY A HACK WHI
    app.run(functools.partial(platform.main, MODEL_DICT["game"]))
