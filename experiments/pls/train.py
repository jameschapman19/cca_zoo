import os
from re import I

import numpy as np
import wandb
from absl import app, flags
from ccagame import pls
from ccagame.datasets.utils import get_training_steps
from ccagame.utils import log_dir
from jax import profiler
from jaxline_fork import platform
from ml_collections import config_flags

"""
So in general flags are things from the command line
When we flags.define_(x) we basically tell python if one of the command line arguments is x then process it
Anything that is defined in the python script gets put into the FLAGS dictionary.
"""

FLAGS = flags.FLAGS
# change the default to your own config file path if you
flags.DEFINE_string(name="model", default="sgha", help="model name")

MODEL_DICT = {
    "game": pls.Game,
    "alphagame": pls.AlphaGame,
    "msg": pls.MSG,
    "power": pls.StochasticPower,
    "incremental": pls.Incremental,
    "sgha": pls.SGHA,
    "gameR":pls.GameR,
    "alphagameR":pls.AlphaGameR
}


def main(argv):
    print(f"MODEL IS {FLAGS.model}")
    # we now need to put some of the stuff from config into
    # config.experiment_kwargs because this is what jaxline
    # gives to our experiment objects
    FLAGS.config.experiment_kwargs = {
        "n_components": FLAGS.config.n_components,
        "num_devices": FLAGS.config.num_devices,
        "data": FLAGS.config.data,
        "batch_size": FLAGS.config.batch_size,
        "learning_rate": FLAGS.config.learning_rate,
        "TV": FLAGS.config.TV,
        "val_interval": FLAGS.config.val_interval,
    }
    FLAGS.config.log_train_data_interval = FLAGS.config.val_interval
    FLAGS.config.log_tensors_interval = FLAGS.config.val_interval
    if FLAGS.config.epochs > 0:
        FLAGS.config.training_steps = get_training_steps(
            FLAGS.config.data, FLAGS.config.epochs, FLAGS.config.batch_size
        )
    os.chdir(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.config.data)
    )
    os.chdir(log_dir())
    FLAGS.config.checkpoint_dir = os.getcwd()
    platform.main(MODEL_DICT[FLAGS.model], argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config",
        help_string="Training configuration file.",
        default=os.getcwd() + "/experiments/pls/config.py",
    )
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    app.run(main)
