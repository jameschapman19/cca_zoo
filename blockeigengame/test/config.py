from jaxline.base_config import get_base_config
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    # get the basic jax config
    config = get_base_config()
    config.random_seed = 42

    # these are given by wandb
    config.learning_rate = 1e-3
    config.num_devices = 1
    config.n_components = 1
    config.batch_size = 128
    config.training_steps = 10
    config.data = "mnist"
    config.validate = False

    # defaults
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = 1
    config.log_train_data_interval = 1
    config.lock()
    return config
