from ccagame.datasets.utils import get_training_steps
from jaxline.base_config import get_base_config
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    # get the basic jax config
    config = get_base_config()

    # these are given by wandb
    config.learning_rate = 5e-3
    config.num_devices = 1
    config.n_components = 16
    config.batch_size = 256
    config.data = "linear"
    config.training_steps = 10000
    config.epochs = None
    if config.epochs is not None:
        config.training_steps = get_training_steps()
    config.val_interval = 50
    config.TCC = True
    config.alpha = False

    # defaults
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = False
    config.interval_type = "steps"
    config.log_tensors_interval = 50
    config.lock()
    return config
