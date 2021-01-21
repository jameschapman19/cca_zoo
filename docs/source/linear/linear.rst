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
and changing its inner_loop argument. We provide a number of inner loops from popular papers in the CCA/PLS literature.

In addition we have wrapped these inner loops into their own classes so that a base user can simply call the outer loop.

.. autoclass:: cca_zoo.wrappers.CCA_Iterative
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

PLS
~~~

.. autoclass:: cca_zoo.wrappers.PLS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   pls = wrappers.PLS(latent_dims=latent_dims, max_iter=max_iter)

   pls.fit(train_view_1, train_view_2)

   pls_results = np.stack(
       (pls.train_correlations[0, 1], pls.predict_corr(test_view_1, test_view_2)[0, 1]))

CCA
~~~

.. autoclass:: cca_zoo.wrappers.CCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
      # %%
      cca = wrappers.CCA(latent_dims=latent_dims, max_iter=max_iter)

      cca.fit(train_view_1, train_view_2)

      cca_results = np.stack(
          (cca.train_correlations[0, 1], cca.predict_corr(test_view_1, test_view_2)[0, 1]))

Sparse CCA by Penalized Matrix Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://academic.oup.com/biostatistics/article/10/3/515/293026

.. autoclass:: cca_zoo.wrappers.PMD
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   pmd = wrappers.PMD(latent_dims=latent_dims, max_iter=max_iter)

   pmd.fit(train_view_1, train_view_2,c=[2,2])

   pmd_results = np.stack(
       (pmd.train_correlations[0, 1], pmd.predict_corr(test_view_1, test_view_2)[0, 1]))

Sparse CCA by Penalization (Parkhomenko)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cca_zoo.wrappers.ParkhomenkoCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   parkhomenko = wrappers.Parkhomenko(latent_dims=latent_dims, max_iter=max_iter)

   parkhomenko.fit(train_view_1, train_view_2,c=[0.0001,0.0001])

   parkhomenko_results = np.stack(
       (parkhomenko.train_correlations[0, 1], parkhomenko.predict_corr(test_view_1, test_view_2)[0, 1]))

Sparse CCA by Rescaled Lasso (Mai)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043?casa_token=pw8OSPmNkzEAAAAA:CcrMA_8g_2po011hQsGQXfiYyvtpBlSS6LJm-z_zANOg6t5YhpFZ-2YJNeCbJdHmT7GXIFZUU7gQl78

.. autoclass:: cca_zoo.wrappers.SCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   scca = wrappers.SCCA(latent_dims=latent_dims, max_iter=max_iter)

   scca.fit(train_view_1, train_view_2,c=[0.0001,0.0001])

   scca_results = np.stack(
       (scca.train_correlations[0, 1], scca.predict_corr(test_view_1, test_view_2)[0, 1]))

Sparse CCA by ADMM (Suo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://arxiv.org/abs/1705.10865

.. autoclass:: cca_zoo.wrappers.SCCA_ADMM
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   scca_admm = wrappers.SCCA_ADMM(latent_dims=latent_dims, max_iter=max_iter)

   scca_admm.fit(train_view_1, train_view_2,c=[0.0001,0.0001])

   scca_admm_results = np.stack(
       (scca_admm.train_correlations[0, 1], scca_admm.predict_corr(test_view_1, test_view_2)[0, 1]))

Elastic CCA by Rescaled Lasso (Waaijenborg)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://pubmed.ncbi.nlm.nih.gov/19689958/

.. autoclass:: cca_zoo.wrappers.ElasticCCA
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   elastic_cca = wrappers.ElasticCCA(latent_dims=latent_dims, max_iter=max_iter)

   elastic_cca.fit(train_view_1, train_view_2,c=[0.0001,0.0001],l1_ratio=[0.1,0.1])

   elastic_cca_results = np.stack(
       (elastic_cca.train_correlations[0, 1], elastic_cca.predict_corr(test_view_1, test_view_2)[0, 1]))

Inner Loops
~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Contents:

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

   kcca.fit(train_view_1, train_view_2, kernel='linear', c=[0.5,0.5])

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

   mcca.fit(train_view_1, train_view_2, c=[0.5,0.5])

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

   gcca.fit(train_view_1, train_view_2, c=[0.5,0.5])

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