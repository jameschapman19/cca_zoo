import gzip
from os.path import join

import pandas as pd


def ukbb_dataset(num_batches, path, batch_size, **kwargs):
    # file naming starts at 1
    batch_ids = list(range(1, num_batches + 1))
    # load one batch to get no. of features and use as holdout data
    # X is brain data
    X = pd.read_csv(join(path, "pack_1_img_sd.tab"), delimiter=" ").to_numpy().T
    f = gzip.GzipFile(join(path, "pack_1_norm.tab.gz"), "r")
    # Y is genetics data
    Y = pd.read_csv(f, delimiter=" ").to_numpy().T
    batch_ids.remove(1)
    return (
        X,
        Y,
        X,
        Y,
        batch_ids,
    )
