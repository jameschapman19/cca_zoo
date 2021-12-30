import os
from re import I

from absl import app, flags
from ccagame import cca
from ccagame.utils import log_dir
from jaxline_fork import platform
from ml_collections import config_flags
import wandb

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="/home/chapmajw/ccagame/experiments/cca/config.py",
)
flags.DEFINE_string(name="model", default="appgrad", help="model name")
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": cca.Game,
    "vicreggame": cca.VicRegGame,
    "genoja": cca.GenOja,
    "sgha": cca.SGHA,
    "appgrad": cca.AppGrad,
}


def main(argv):
    print(f"MODEL IS {FLAGS.model}")
    if FLAGS.config.data=='mnist':
        FLAGS.config.training_steps=2#int(FLAGS.config.epochs*60000/FLAGS.config.batch_size)
    # we now need to put some of the stuff from config into
    # config.experiment_kwargs because this is what jaxline
    # gives to our experiment objects
    FLAGS.config.experiment_kwargs = {
        "n_components": FLAGS.config.n_components,
        "num_devices": FLAGS.config.num_devices,
        "data": FLAGS.config.data,
        "batch_size": FLAGS.config.batch_size,
        "learning_rate": FLAGS.config.learning_rate,
        "validate":FLAGS.config.validate,
        "TV":FLAGS.config.TV,
    }
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),FLAGS.config.data))
    os.chdir(log_dir())
    FLAGS.config.checkpoint_dir = os.getcwd()
    platform.main(MODEL_DICT[FLAGS.model], argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    app.run(main)
