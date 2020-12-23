.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Linear Models
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Intro
-----

Base
----

.. automodule:: cca_zoo.wrappers.CCA_Base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Alternating Least Squares Methods
----------

.. automodule:: cca_zoo.wrappers.CCA_ALS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Kernel CCA
----------

.. automodule:: cca_zoo.wrappers.KCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Multiset CCA
-----------------

.. automodule:: cca_zoo.wrappers.MCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Generalized CCA
-----------------

.. automodule:: cca_zoo.wrappers.GCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Grid search cross-validation
----------------------------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   c1 = [3, 7, 9]
   c2 = [3, 7, 9]
   param_candidates = {'c': list(itertools.product(c1, c2))}

   pmd = cca_zoo.wrappers.CCA_ALS(latent_dims=latent_dims, method='pmd').gridsearch_fit(train_set_1, train_set_2,
                                                                                              param_candidates=param_candidates,folds=cv_folds,verbose=True)