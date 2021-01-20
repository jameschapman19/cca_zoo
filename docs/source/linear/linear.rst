.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Linear (and Kernel) Models
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Intro
-----

Within cca-zoo, linear and kernel methods inherit a common base structure. This gives shared functionality and a standard
API that is somewhat similar to the cross_decomposition module in scikit-learn with familiar methods fit(), transform()
and fit_transform().


Base
----
The base class defines the shared structure of each of the models implemented in wrappers.py.

The idea is that new methods which inherit the base class can benefit from general functions such as demean_data and gridsearch_fit provided that

.. autoclass:: cca_zoo.wrappers.CCA_Base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Iterative Methods
-----------------

The CCA_Iterative class gives further flexibility to use iterative optimization methods.

In it's basic form, CCA_Iterative performs unregularized PLS but by inheriting this class
and changing its inner_loop argument.

.. autoclass:: cca_zoo.wrappers.CCA_Iterative
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. toctree::
   :maxdepth: 3

   innerloop.rst



Kernel CCA
----------

The KCCA class allows the use of linear, polynomial and rbf kernels (and their associated parameters).

.. autoclass:: cca_zoo.wrappers.KCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Example
~~~~~~~

.. sourcecode:: python

   from cca_zoo import wrappers
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   kcca = wrappers.KCCA(latent_dims=latent_dims)
   # small ammount of regularisation added since data is not full rank
   params = {'kernel': 'linear','c': [0.5, 0.5]}

   kcca.fit(train_view_1, train_view_2, params=params)

   kcca_results = np.stack((kcca.train_correlations[0, 1], kcca.predict_corr(test_view_1, test_view_2)[0, 1]))

Multiset CCA
-------------

Multiset CCA is implemented as a generalized eigenvalue problem.

.. autoclass:: cca_zoo.wrappers.MCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Example
~~~~~~~

.. sourcecode:: python

   from cca_zoo import wrappers
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   mcca = wrappers.MCCA(latent_dims=latent_dims)
   # small ammount of regularisation added since data is not full rank
   params = {'c': [0.5, 0.5]}

   mcca.fit(train_view_1, train_view_2, params=params)

   mcca_results = np.stack((mcca.train_correlations[0, 1], mcca.predict_corr(test_view_1, test_view_2)[0, 1]))

Generalized CCA
-----------------

Generalized CCA is implemented as a generalized eigenvalue problem.

.. autoclass:: cca_zoo.wrappers.GCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Example
~~~~~~~

.. sourcecode:: python

   from cca_zoo import wrappers
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   gcca = wrappers.GCCA(latent_dims=latent_dims)
   # regularisation from 0<'c'<1 for each view
   params = {'c': [1, 1]}

   gcca.fit(train_view_1, train_view_2, params=params)

   gcca_results = np.stack((gcca.train_correlations[0, 1], gcca.predict_corr(test_view_1, test_view_2)[0, 1]))

Grid search cross-validation
----------------------------

All of the models inherit gridsearch_fit() from the base class. This can be used to select from a user defined parameter grid.

Example
~~~~~~~

.. sourcecode:: python

   from cca_zoo import wrappers
   # PMD
   c1 = [1, 3, 7, 9]
   c2 = [1, 3, 7, 9]
   param_candidates = {'c': list(itertools.product(c1, c2))}

   pmd = wrappers.PMD(latent_dims=latent_dims, tol=1e-5, max_iter=max_iter).gridsearch_fit(
       train_view_1,
       train_view_2,
       param_candidates=param_candidates,
       folds=cv_folds,
       verbose=True, jobs=jobs,
       plot=True)

   pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_view_1, test_view_2)[0, 1, :]))