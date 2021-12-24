#from ccagame import pls
#from experiments.experiment_config import LEARNING_RATE
import functools
from os import environ
from absl import app, flags
#from ccagame.utils import data_stream_UKBB, log_dir
#from datasets.mnist import mnist
from jaxline_fork import platform
import jax.numpy as jnp
import os
#from experiments import parse_args, get_config
import gzip 
import pandas as pd
import time 
import numpy as np
# Right so basically this should run from command line/bash script
# mnist.py --cores 4 --n_components 4 --batch_size 16 --lr 0.001 --model game
#MODEL_DICT = {
#    "game": pls.Game,
#    "msg": pls.MSG,
#    "oja": pls.Oja,
#    "power": pls.StochasticPower,
#    "incremental":pls.Incremental
#}

# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
   # args = parse_args()
   # environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices}"
   # FLAGS = flags.FLAGS
    PATH = '../ccagame_data'
    batch_ids =[2, 1, 3,4,5,6]
    batch_ids.remove(1)
    num = len(batch_ids)
    batch_size = 1
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(0)
    print(rng)
    perm = rng.permutation(batch_ids)
    print(perm)
    for i in range(num_batches):
        batch_idx = perm[i]
        #load batch - batches are in groups of 500 subjects
        print(batch_idx)
    exit()
    #TEST OUT LOADING UKBB DATA 
    input_data_iterator = data_stream_UKBB(2, PATH, batch_size=args.batch_size)
    #correct_U, correct_S, correct_V = jnp.linalg.svd(X.T @ Y)
    #dof=X.shape[0]
    #print(f"TV : {correct_S[:args.n_components].sum()/dof}")
    #correct_U = correct_U[:, :args.n_components]
    #correct_V = correct_V[:args.n_components, :].T
    FLAGS.config = get_config(
        input_data_iterator,
        dims=[400, 384],
        num_devices=args.devices,
        n_components=args.n_components,
        training_steps=args.training_steps,
        #correct_eigenvectors=[correct_U,correct_V],
        learning_rate=args.learning_rate,
        model=args.model,
        batch_size=args.batch_size,
        #holdout=[X_te,Y_te]
    )
    flags.mark_flag_as_required("config")
    #magic function which does what pytorch-lightning does which is to make a new numbered version in the directory for each run
    os.chdir(log_dir())
    #TODO THIS IS CURRENTLY A HACK WHI
    app.run(functools.partial(platform.main, MODEL_DICT['game']))
