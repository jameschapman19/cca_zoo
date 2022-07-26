import functools

import wandb
from absl import app, flags
from jax.profiler import trace
from jaxline import platform

from blockeigengame import cca, pls, rcca

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "batch size")
_MODEL = flags.DEFINE_string("model", "msg", "model")
_DATA = flags.DEFINE_string("data", "linear", "dataset name")
_EXPERIMENT = flags.DEFINE_string(
    "experiment", "CCA", "whether to run a PLS or CCA experiment"
)
_N_COMPONENTS = flags.DEFINE_integer("n_components", 4, "number of components")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-1, "learning rate")
_EPOCHS = flags.DEFINE_integer("epochs", 1, "epochs")
_LOGGING_INTERVAL = flags.DEFINE_float("logging_interval", 0.01, "logging interval")

MODEL_DICT = {
    "CCA": {
        "game": cca.Game,
        "genoja": cca.GenOja,
        "sgha": cca.SGHA,
        "appgrad": cca.AppGrad,
        "ssgd": cca.SSGD,
        "msg": cca.MSG,
        "rgame": rcca.RGame,
        "saa": cca.SAA,
        "alphagame": cca.AlphaGame,
        "mgame": cca.MGame,
        "incremental": cca.Incremental,
        "elasticgame": cca.ElasticGame,
    },
    "PLS": {
        "game": pls.Game,
        "msg": pls.MSG,
        "power": pls.StochasticPower,
        "incremental": pls.Incremental,
        "sgha": pls.SGHA,
        "ssgd": pls.SSGD,
        "saa": pls.SAA,
        "alphagame": pls.AlphaGame,
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
    "logging_interval": _LOGGING_INTERVAL.default,
}

if __name__ == "__main__":
    wandb.init(config=defaults, sync_tensorboard=True)
    Experiment = MODEL_DICT[wandb.config.experiment][wandb.config.model]
    #with trace("/tmp/jax-trace", create_perfetto_link=True):
    app.run(functools.partial(platform.main, Experiment))
