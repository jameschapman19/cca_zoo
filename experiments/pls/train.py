import functools
import os
import sys
from re import I

import wandb
from absl import app, flags
from blockeigengame import pls
from jaxline import base_config, platform
from ml_collections import config_flags


MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "power": pls.StochasticPower,
    "incremental": pls.Incremental,
    "sgha": pls.SGHA,
}


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    Experiment = MODEL_DICT["msg"]
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, Experiment))