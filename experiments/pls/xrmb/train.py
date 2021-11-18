from ccagame import pls
from ccagame.pca.game import Game
from ccagame.pca.oja import Oja
from ccagame.pls.stochasticpower import StochasticPower
from ccagame.pls.msg import MSG
from jaxline import platform
import functools
from os import environ
from absl import app, flags
from ccagame.utils import data_stream
from datasets.xrmb import xrmb
from jaxline import platform
from jaxline.base_config import get_base_config
import argparse

#Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game

#These are the defaults for the above arguments
CORES=4
N_COMPONENTS=4
BATCH_SIZE=None
LR=1e-3
MODEL='game'
#This is used to turn name of model on command line into model class
MODEL_DICT={
    'game':Game,
    'msg':MSG,
    'oja':Oja,
    'power':StochasticPower,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=CORES)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=N_COMPONENTS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--model", type=float, default=MODEL)
    return parser.parse_args()

#THIS IS ALL OF THE PARAMETERS OF JAXLINE EXPERIMENT
#ITS BASICALLY A DICTIONARY
def get_config(
    data,
    devices,
    dims,
    k_per_device=1,
    log_tensors_interval=1,
    log_train_data_interval=1,
    training_steps=100,
):
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "k_per_device": k_per_device,
        "num_devices": devices,
        "dims": dims,
        "data": data,
    }
    config.training_steps = training_steps
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = log_tensors_interval
    config.log_train_data_interval = log_train_data_interval
    config.lock()
    return config



# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    args = parse_args()
    environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices}"
    FLAGS = flags.FLAGS
    view_1_tr, view_2_tr, view_1_te, view_2_te = xrmb()
    input_data_iterator = data_stream(view_1_tr, Y=view_2_tr, batch_size=args.batch_size)
    FLAGS.config = get_config(input_data_iterator, [view_1_tr.shape[1],view_2_tr.shape[1]],args.cores,n_components=args.n_components)
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, MODEL_DICT[args.model]))