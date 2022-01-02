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
    if FLAGS.config.data == "mnist":
        FLAGS.config.training_steps = int(
            FLAGS.config.epochs * 60000 / FLAGS.config.batch_size
        )
    # we now need to put some of the stuff from config into
    # config.experiment_kwargs because this is what jaxline
    # gives to our experiment objects
    FLAGS.config.experiment_kwargs = {
        "n_components": FLAGS.config.n_components,
        "num_devices": FLAGS.config.num_devices,
        "data": FLAGS.config.data,
        "batch_size": FLAGS.config.batch_size,
        "learning_rate": FLAGS.config.learning_rate,
        "validate": FLAGS.config.validate,
        "TV": FLAGS.config.TV,
        "alpha":FLAGS.config.alpha
    }
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
