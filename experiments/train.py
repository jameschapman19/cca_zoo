import functools
import os
from re import I

import wandb
from absl import app, flags
from blockeigengame import cca, pls, rcca
from blockeigengame._utils import log_dir
from jaxline import platform

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "batch size")
_MODEL = flags.DEFINE_string("model", "game", "model")
_DATA = flags.DEFINE_string("data", "mediamill", "dataset name")
_EXPERIMENT = flags.DEFINE_string(
    "experiment", "CCA", "whether to run a PLS or CCA experiment"
)
_N_COMPONENTS = flags.DEFINE_integer("n_components", 4, "number of components")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-1, "learning rate")
_EPOCHS = flags.DEFINE_integer("epochs", 1, "epochs")

MODEL_DICT = {
    "CCA": {
        "game": cca.Game,
        "genoja": cca.GenOja,
        "sgha": cca.SGHA,
        "appgrad": cca.AppGrad,
        "ssgd": cca.SSGD,
        "msg": cca.MSG,
        "rcca": rcca.Game,
        "saa": cca.SAA,
    },
    "PLS": {
        "game": pls.Game,
        "msg": pls.MSG,
        "power": pls.StochasticPower,
        "incremental": pls.Incremental,
        "sgha": pls.SGHA,
        "ssgd": pls.SSGD,
        "saa": pls.SAA,
    },
}

defaults = {
    "experiment": _EXPERIMENT.default,
    "model": _MODEL.default,
    "data": _DATA.default,
    "epochs": _EPOCHS.default,
    "batch_size": _BATCH_SIZE.default,
    "n_components": _N_COMPONENTS.default,
    "learning_rate": _LEARNING_RATE.default,
}


if __name__ == "__main__":
    wandb.init(config=defaults, sync_tensorboard=True)
    Experiment = MODEL_DICT[wandb.config.experiment][wandb.config.model]
    app.run(functools.partial(platform.main, Experiment))
