.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Canonically Correlated Autoencoders
===========

.. sourcecode:: python

   import cca_zoo
   cfg = Config()
   cfg.method = cca_zoo.dvcca.DCCAE
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   dcca = cca_zoo.deepwrapper.DeepWrapper(cfg)
   dcca.fit(train_set_1, train_set_2)

Intro
-----

.. automodule:: cca_zoo.dccae
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource