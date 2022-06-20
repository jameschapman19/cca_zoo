import os
from re import I
from jaxline import platform
from absl import app, flags
from blockeigengame import cca
from ml_collections import config_flags
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string(name="model", default="sghagame", help="model name")
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
MODEL_DICT = {
    "game": cca.Game,
    "genoja": cca.GenOja,
    "sgha": cca.SGHA,
    "appgrad": cca.AppGrad,
    "ssgd": cca.SSGD,
    "ssgdgame": cca.SSGDGame,
    "sghagame": cca.SGHAGame,
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
        "TCC": FLAGS.config.TCC,
        "alpha": FLAGS.config.alpha,
        "val_interval": FLAGS.config.val_interval,
        "random_state": FLAGS.config.random_seed,
    }
    FLAGS.config.log_train_data_interval = FLAGS.config.val_interval
    FLAGS.config.log_tensors_interval = FLAGS.config.val_interval
    os.chdir(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.config.data)
    )
    #os.chdir(log_dir())
    FLAGS.config.checkpoint_dir = os.getcwd()
    platform.main(MODEL_DICT[FLAGS.model], argv)


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
t    config_flags.DEFINE_config_file(
        "config",
        help_string="Training configuration file.",
        default=os.getcwd() + "/experiments/cca/config.py",
    )
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    app.run(main)
