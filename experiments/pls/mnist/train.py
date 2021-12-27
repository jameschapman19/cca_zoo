from re import I
import os
from ccagame import pls
from absl import app, flags
from ccagame.utils import log_dir
from jaxline_fork import platform
import os
import wandb
from absl import flags
from ml_collections import config_flags


"""
So in general flags are things from the command line
When we flags.define_(x) we basically tell python if one of the command line arguments is x then process it
Anything that is defined in the python script gets put into the FLAGS dictionary.
"""

FLAGS = flags.FLAGS
# change the default to your own config file path if you
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="/home/chapmajw/ccagame/experiments/pls/mnist/config.py",
)
flags.DEFINE_string(name="model", default="game", help="model name")

MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "oja": pls.Oja,
    "power": pls.StochasticPower,
    "incremental": pls.Incremental,
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
    }
    platform.main(MODEL_DICT[FLAGS.model], argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    os.chdir(log_dir())
    # wandb gets initialized now because wandb generates command line arguments which are used by flags
    wandb.init(sync_tensorboard=True)
    # app basically tells flags to process things that are defined now and run main
    app.run(main)
