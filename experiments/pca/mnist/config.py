from jaxline.base_config import get_base_config
from ccagame.utils import data_stream
from datasets.mnist import mnist
import jax.numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    #get the basic jax config
    config = get_base_config()

    """
    Data section: this is what we change to use different data
    """
    X, _, X_te, _ = mnist()
    """
    End of data section
    """

    #these are given by wandb
    config.learning_rate=1e-3
    config.num_devices=1
    config.n_components=4
    config.batch_size=0
    config.training_steps = 10000

    """
    This shouldn't need to be changed
    """
    input_data_iterator = data_stream(
        X, batch_size=config.batch_size
    )
    correct_U, _,_ = jnp.linalg.svd(X.T @ X)
    correct_U = correct_U[:, : config.n_components]
    config.experiment_kwargs = {
        "n_components": config.n_components,
        "num_devices": config.num_devices,
        "dims": X.shape[1],
        "data": input_data_iterator,
        "correct_eigenvectors": correct_U,
        "batch_size":config.batch_size,
        "learning_rate":config.learning_rate,
        "holdout":[X_te]
    }
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = 1
    config.log_train_data_interval = 1
    config.lock()
    return config