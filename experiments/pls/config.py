from jaxline.base_config import get_base_config
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    # get the basic jax config
    config = get_base_config()
    config.random_seed = 21

    # these are given by wandb
    config.learning_rate = 1e-2
    config.num_devices = 1
    config.n_components = 8
    config.batch_size = 1
    config.data = "linear"
    config.training_steps = 10000
    config.epochs = 1
    config.val_interval = 50
    config.TV = True

    # defaults
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = False
    config.interval_type = "steps"
    config.log_train_data_interval = 50
    config.log_tensors_interval = 50
    config.lock()
    return config
