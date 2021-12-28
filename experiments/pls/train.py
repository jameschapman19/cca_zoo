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
from jax import profiler

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
flags.DEFINE_string(name="model", default="incremental", help="model name")

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
        "validate":FLAGS.config.validate
    }
    os.chdir(FLAGS.config.data)
    os.chdir(log_dir())
    profiler.start_trace("tmp")
    platform.main(MODEL_DICT[FLAGS.model], argv)
    profiler.stop_trace()


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    # environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={config.devices}"
    app.run(main)
