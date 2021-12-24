from jaxline.base_config import get_base_config
from ccagame.utils import data_stream
from datasets.mnist import mnist
import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    # get the basic jax config
    config = get_base_config()

    # these are given by wandb
    config.learning_rate = 1e-3
    config.num_devices = 1
    config.n_components = 4
    config.batch_size = 1
    config.training_steps = 10
    config.data = "ukbb"
    """
    This shouldn't need to be changed
    """
    config.experiment_kwargs = {
        "n_components": config.n_components,
        "num_devices": config.num_devices,
        "data": "ukbb",
        "path": "/mnt/c/Users/anala/Documents/PhD/ccagame_data",
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = 1
    config.log_train_data_interval = 1
    config.lock()
    return config
