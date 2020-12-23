.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Linear (and Kernel) Models
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Intro
-----

Base
----
The base class defines the shared structure of each of the models implemented in wrappers.py.

.. autoclass:: cca_zoo.wrappers.CCA_Base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Alternating Least Squares Methods
----------

.. autoclass:: cca_zoo.wrappers.CCA_ALS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Kernel CCA
----------

.. autoclass:: cca_zoo.wrappers.KCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Multiset CCA
-----------------

.. autoclass:: cca_zoo.wrappers.MCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Generalized CCA
-----------------

.. autoclass:: cca_zoo.wrappers.GCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Grid search cross-validation
----------------------------

.. sourcecode:: python

   from cca_zoo import wrappers
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   c1 = [3, 7, 9]
   c2 = [3, 7, 9]
   param_candidates = {'c': list(itertools.product(c1, c2))}

   pmd = wrappers.CCA_ALS(latent_dims=latent_dims, method='pmd').gridsearch_fit(train_set_1, train_set_2,
                                                                                              param_candidates=param_candidates,folds=cv_folds,verbose=True)