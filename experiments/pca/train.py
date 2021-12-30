from re import I
import os
from ccagame import pca
from absl import flags, app
from ccagame.utils import log_dir
import os
import wandb
from jaxline_fork import platform
from absl import flags
from ml_collections import config_flags
FLAGS = flags.FLAGS
flags.DEFINE_string(name="model", default="game", help="model name")
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": pca.Game,
    "oja": pca.Oja,
    "gha": pca.GHA,
}


def main(argv):
    print(f"MODEL IS {FLAGS.model}")
    if FLAGS.config.data=='mnist':
        FLAGS.config.training_steps=int(FLAGS.config.epochs*60000/FLAGS.config.batch_size)
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


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default=os.getcwd()+"/experiments/pca/config.py",
    )
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    app.run(main)
