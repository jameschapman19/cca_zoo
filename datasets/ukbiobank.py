from os.path import join

import pandas as pd


# TODO
def ukbiobank(path):
    """
    Download, parse and process UKBiobank data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2, test_view_1, test_view_2 = datasets.ukbiobank()

    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    brain_train = pd.read_csv(join(path, 'Epilepsy_MRI_train_zscores.csv'), header=None).to_numpy()
    genetics_train = pd.read_csv(join(path, 'Epilepsy_genetics_train_processed.csv'), header=None).to_numpy()
    brain_test = pd.read_csv(join(path, 'Epilepsy_MRI_test_zscores.csv'), header=None).to_numpy()
    genetics_test = pd.read_csv(join(path, 'Epilepsy_genetics_test_processed.csv'), header=None).to_numpy()

    return brain_train, genetics_train, brain_test, genetics_test
