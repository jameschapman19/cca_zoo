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
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", help_string="Training configuration file.", default="./experiments/pls/config_ukbb.py")
flags.DEFINE_string(name="model", default="game", help="model name")
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "oja": pls.Oja,
    "power": pls.StochasticPower,
    "incremental": pls.Incremental,
}


def main(argv):
    print(f"MODEL IS {FLAGS.model}")
    #we now need to put some of the stuff from config into 
    # config.experiment_kwargs because this is what jaxline 
    # gives to our experiment objects
    FLAGS.config.experiment_kwargs = {
        "n_components": FLAGS.config.n_components,
        "num_devices": FLAGS.config.num_devices,
        "data": FLAGS.config.data,
        "num_batches": FLAGS.config.num_batches, 
        "path": FLAGS.config.path,
        "batch_size":FLAGS.config.batch_size,
        "learning_rate":FLAGS.config.learning_rate,
    }
    platform.main(MODEL_DICT[FLAGS.model],argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    #wandb.init(sync_tensorboard=True)
    #wandb_config = wandb.config
    #print(wandb_config)
    # environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={config.devices}"
    # magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    os.chdir(log_dir())
    app.run(main)
