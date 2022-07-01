import functools
from re import I
import wandb
from absl import app
from blockeigengame import cca, pls, rcca
from jaxline import platform
from absl import flags
import os
from blockeigengame._utils import log_dir
flags.DEFINE_integer('batch_size',0,'batch size')
flags.DEFINE_string('data','linear','dataset name')
flags.DEFINE_string('experiment','CCA','whether to run a PLS or CCA experiment')
flags.DEFINE_integer('training_steps',1000,'training steps')
flags.DEFINE_integer('n_components',3,'number of components')
flags.DEFINE_float('learning_rate',1e-3,'number of components')

MODEL_DICT = {
    "CCA": {
        "game": cca.Game,
        "genoja": cca.GenOja,
        "sgha": cca.SGHA,
        "appgrad": cca.AppGrad,
        "ssgd": cca.SSGD,
        "msg": cca.MSG,
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

defaults={
    'experiment':'CCA',
    'model':'ssgd',
    'data':'linear'
}


if __name__ == "__main__":
    wandb.init(config=defaults)
    Experiment = MODEL_DICT[wandb.config.experiment][wandb.config.model]
    app.run(functools.partial(platform.main, Experiment))
