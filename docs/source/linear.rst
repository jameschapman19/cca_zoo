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


Linear CCA
----------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims)
   linear_cca.fit(train_set_1, train_set_2)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims,method='scikit)
   linear_cca.fit(train_set_1, train_set_2)

Sparse CCA
----------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims,method='pmd')
   linear_cca.fit(train_set_1, train_set_2)

 .. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims,method='parkhomenko')
   linear_cca.fit(train_set_1, train_set_2)

 .. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='scca')
   linear_cca.fit(train_set_1, train_set_2)


Kernel CCA
----------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method'kernel')
   linear_cca.fit(train_set_1, train_set_2)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method'kernel')
   linear_cca.fit(train_set_1, train_set_2)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method'kernel')
   linear_cca.fit(train_set_1, train_set_2)

Multiple view CCA
-----------------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims)
   linear_cca.fit(train_set_1, train_set_2, train_set_3)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gep')
   linear_cca.fit(train_set_1, train_set_2, train_set_3)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='mcca')
   linear_cca.fit(train_set_1, train_set_2, train_set_3)

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gcca')
   linear_cca.fit(train_set_1, train_set_2, train_set_3)

Grid search cross-validation
----------------------------

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   c1 = [3, 7, 9]
   c2 = [3, 7, 9]
   param_candidates = {'c': list(itertools.product(c1, c2))}

   pmd = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='pmd').gridsearch_fit(train_set_1, train_set_2,
                                                                                              param_candidates=param_candidates,
                                                                                              folds=cv_folds,
                                                                                              verbose=True)