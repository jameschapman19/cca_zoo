import functools
import os
import sys
from re import I

import jax
import wandb
from absl import app, flags
from blockeigengame import cca, pls, rcca
from blockeigengame._utils import log_dir
from jaxline import base_config, platform
from ml_collections import config_flags

MODEL_DICT = {
    "CCA": {
        "game": cca.Game,
        "genoja": cca.GenOja,
        "sgha": cca.SGHA,
        "appgrad": cca.AppGrad,
        "ssgd": cca.SSGD,
        "msg": cca.MSG,
        "oldgame": cca.OldGame,
        "rcca":rcca.Game,
    },
    "PLS": {
        "game": pls.Game,
        "msg": pls.MSG,
        "power": pls.StochasticPower,
        "incremental": pls.Incremental,
        "sgha": pls.SGHA,
        "ssgd": pls.SSGD,
    },
}

# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    wandb.init(sync_tensorboard=True)
    wandb_config = wandb.config
    Experiment = MODEL_DICT["CCA"]["rcca"]
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, Experiment))
