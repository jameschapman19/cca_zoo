from jaxline import base_config
from ml_collections import config_dict

N_TRAIN_EXAMPLES=1000

def get_training_steps(batch_size, n_epochs):
    if batch_size==0:
        batch_size=N_TRAIN_EXAMPLES
    return (N_TRAIN_EXAMPLES * n_epochs) // batch_size

def get_config():
    config = base_config.get_base_config()
    # these are given by wandb
    config.learning_rate = 1e-2
    config.num_devices = 1
    config.n_components = 4
    config.batch_size = 16
    config.epochs = 10000
    config.data = "linear"
    config.training_steps = get_training_steps(config.batch_size, config.epochs)
    config.TCC = True
    config.alpha = False
    config.beta0=1e-3

    # defaults
    config.checkpoint_dir = "jaxlog"
    config.interval_type = "steps"
    config.log_tensors_interval = 10
    config.experiment_kwargs.config = {
            "model": 'pls',
            "n_components": config.n_components,
            "num_devices": config.num_devices,
            "data": config.data,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "TCC": config.TCC,
            "alpha": config.alpha,
            "beta0": config.beta0,
            "c":5e-3,
            "random_state": config.random_seed,
            "whitening_batch_size":10 * config.n_components
        }
    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    config.lock()
    return config