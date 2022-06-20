from jaxline.base_config import get_base_config
from ml_collections import config_dict

N_TRAIN_EXAMPLES=1000

def get_training_steps(batch_size, n_epochs):
    return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config() -> config_dict.ConfigDict:
    # get the basic jax config
    config = get_base_config()

    # these are given by wandb
    config.learning_rate = 1e-6
    config.num_devices = 1
    config.n_components = 1
    config.batch_size = 256
    config.epochs = 10000
    config.data = "mnist"
    config.training_steps = get_training_steps(config.batch_size, config.epochs)
    config.val_interval = 1
    config.TCC = True
    config.alpha = False

    # defaults
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = False
    config.interval_type = "steps"
    config.log_train_data_interval = 50
    config.log_tensors_interval = 50
    config.lock()
    return config
