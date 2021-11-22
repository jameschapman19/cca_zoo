from jaxline.base_config import get_base_config
import argparse

# These are the defaults for the above arguments
DEVICES = 1
N_COMPONENTS = 16
LEARNING_RATE = 1e-3
MODEL = "game"
BATCH_SIZE = 0
# This is used to turn name of model on command line into model class
TRAINING_STEPS = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=DEVICES)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--training_steps", type=int, default=TRAINING_STEPS)
    return parser.parse_args()


# THIS IS ALL OF THE PARAMETERS OF JAXLINE EXPERIMENT
# ITS BASICALLY A DICTIONARY
def get_config(
    data,
    dims=None,
    num_devices=1,
    n_components=1,
    log_tensors_interval=1,
    log_train_data_interval=1,
    training_steps=100,
    correct_eigenvectors=None,
    model=None,
    **kwargs
):
    """Return config object for training."""
    config = get_base_config()
    config.experiment_kwargs = {
        "n_components": n_components,
        "num_devices": num_devices,
        "dims": dims,
        "data": data,
        "model": model,
        "correct_eigenvectors": correct_eigenvectors,
        **kwargs,
    }
    config.training_steps = training_steps
    config.checkpoint_dir = "jaxlog"
    config.train_checkpoint_all_hosts = True
    config.log_tensors_interval = log_tensors_interval
    config.log_train_data_interval = log_train_data_interval
    config.lock()
    return config
