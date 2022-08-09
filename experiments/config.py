import os

import wandb
from jaxline.base_config import get_base_config

from blockeigengame._utils import log_dir

N_TRAIN_EXAMPLES = {
    "linear": 10000,
    "exponential": 10000,
    "mnist": 60000,
    "mediamill": 10000,
    "xrmb": 1000000,
}


def get_training_steps(batch_size, n_epochs, data):
    if batch_size == 0:
        batch_size = N_TRAIN_EXAMPLES[data]
    return (N_TRAIN_EXAMPLES[data] * n_epochs) // batch_size


def get_config():
    config = get_base_config()
    # defaults
    config.interval_type = "steps"
    if wandb.config.get("batch_size", 0)==0:
        batch_size=N_TRAIN_EXAMPLES[wandb.config.get("data", "linear")]
    else:
        batch_size=wandb.config.get("batch_size", 0)
    config.log_tensors_interval = int(wandb.config.get("logging_interval", 1) * N_TRAIN_EXAMPLES[
        wandb.config.get("data", "linear")] // batch_size)
    config.log_train_data_interval = config.log_tensors_interval
    config.save_checkpoint_interval = config.log_tensors_interval
    # If True, asynchronously logs training data from every training step.
    config.log_all_train_data = True
    config.experiment_kwargs.config = {
        "model": "cca",
        "n_components": wandb.config.get("n_components", 3),
        "num_devices": wandb.config.get("num_devices", 1),
        "data": wandb.config.get("data", "linear"),
        "batch_size": batch_size,
        "learning_rate": wandb.config.get("learning_rate", 1e-2),
        "alpha": wandb.config.get("alpha", 1e-3),
        "beta0": wandb.config.get("beta0", 1e-3),
        "c": wandb.config.get("c", [0, 0]),
        "tau": wandb.config.get("tau", [0.5, 0.5]),
        "random_state": wandb.config.get("random_seed", 0),
        "whitening_batch_size": 10 * wandb.config.get("n_components", 1),
        "riemann": False,
        "val_interval": config.log_tensors_interval,
        "kappa": 4,
    }
    config.epochs = wandb.config.get("epochs", 1)
    config.training_steps = get_training_steps(
        config.experiment_kwargs.config["batch_size"],
        config.epochs,
        wandb.config.get("data", "linear"),
    )
    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    exp_dir = os.path.join("results", wandb.config.experiment, wandb.config.get("data", "linear"))
    os.makedirs(exp_dir, exist_ok=True)
    os.chdir(exp_dir)
    os.chdir(log_dir())
    config.checkpoint_dir = os.getcwd()
    config.lock()
    return config
