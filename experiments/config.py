import wandb
from jaxline.base_config import get_base_config
from blockeigengame._utils import log_dir
import os
N_TRAIN_EXAMPLES = 1000


def get_training_steps(batch_size, n_epochs):
    if batch_size == 0:
        batch_size = N_TRAIN_EXAMPLES
    return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config():
    config=get_base_config()
    # defaults
    config.checkpoint_dir = "jaxlog"
    config.interval_type = "steps"
    config.log_tensors_interval = 10
    config.experiment_kwargs.config = {
        "model": "cca",
        "n_components": wandb.config.get('n_components',3),
        "num_devices": wandb.config.get('num_devices',1),
        "data": wandb.config.get('data','linear'),
        "batch_size": wandb.config.get('batch_size',0),
        "learning_rate": wandb.config.get('learning_rate',1e-2),
        "alpha": wandb.config.get('alpha',False),
        "beta0": wandb.config.get('beta0',1e-3),
        "c": wandb.config.get('c',5e-3),
        "tau": wandb.config.get('tau',[0.7, 0.7]),
        "random_state": wandb.config.get('random_seed',42),
        "whitening_batch_size": 10 * wandb.config.get('n_components',1),
        "riemann": False,
        "val_interval": 10,
    }
    config.epochs = wandb.config.get('epochs',1000)
    config.training_steps = get_training_steps(config.experiment_kwargs.config['batch_size'], config.epochs)
    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    exp_dir=os.path.join('experiments',wandb.config.experiment,wandb.config.data)
    os.makedirs(
        exp_dir,exist_ok=True)
    os.chdir(exp_dir)
    os.chdir(log_dir())
    config.checkpoint_dir=os.getcwd()
    config.lock()
    return config
