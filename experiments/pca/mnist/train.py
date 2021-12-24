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
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="/home/chapmajw/ccagame/experiments/pca/mnist/config.py",
)
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
    FLAGS.config.experiment_kwargs = {
        "n_components": FLAGS.config.n_components,
        "num_devices": FLAGS.config.num_devices,
        "data": "mnist",
        "batch_size": FLAGS.config.batch_size,
        "learning_rate": FLAGS.config.learning_rate,
    }
    platform.main(MODEL_DICT[FLAGS.model], argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    os.chdir(log_dir())
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    # environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={config.devices}"
    # magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    app.run(main)
