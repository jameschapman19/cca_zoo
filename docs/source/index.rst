.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cca-zoo's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   linear/linear
   deep/deep

cca-zoo
========

cca-zoo is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data.

Where possible I have followed the scikit-learn/mvlearn APIs and models therefore have fit/transform/fit_transform.

Models can be tuned using a gridsearch.

We provide a tutorial notebook hosted on google colab: https://colab.research.google.com/drive/1reldEBw69hsOtwQOvYsbGGnpvH__b7WF?usp=sharing

Look how easy it is to use:

.. sourcecode:: python

   from cca_zoo import wrappers
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   # %%
   linear_cca = wrappers.CCA_ITER(latent_dims=latent_dims,max_iter=max_iter)
   linear_cca.fit(train_view_1, train_view_2)

Features
--------

Linear CCA/PLS:
~~~~~~~~~~~~~~~

A variety of linear CCA and PLS methods implemented where possible as the solutions to generalized eigenvalue problems and otherwise using alternating minimization methods for non-convex optimisation based on least squares

GCCA (Generalized CCA):
~~~~~~~~~~~~~~~~~~~~~~~

https://academic.oup.com/biomet/article-abstract/58/3/433/233349?redirectedFrom=fulltext

MCCA (Multiset CCA):
~~~~~~~~~~~~~~~~~~~~

SCCA (Sparse CCA - Mai):
~~~~~~~~~~~~~~~~~~~~~~~~

https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043?casa_token=pw8OSPmNkzEAAAAA:CcrMA_8g_2po011hQsGQXfiYyvtpBlSS6LJm-z_zANOg6t5YhpFZ-2YJNeCbJdHmT7GXIFZUU7gQl78

PMD (Sparse PLS/PMD/Penalized Matrix Decomposition - Witten):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://academic.oup.com/biostatistics/article/10/3/515/293026

PCCA (Penalized CCA - elastic net - Waaijenborg):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://pubmed.ncbi.nlm.nih.gov/19689958/

SCCA_ADMM (Sparse canonical correlation analysis-Suo):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://arxiv.org/abs/1705.10865

Kernel CCA
~~~~~~~~~~

CCA solved using the kernel method.

Linear Kernel
RBF Kernel
Polynomial Kernels

DCCA (Deep CCA):
~~~~~~~~~~~~~~~~

https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
https://arxiv.org/pdf/1510.02054v1.pdf
Using either Andrew's original Tracenorm Objective or Wang's alternating least squares solution

DGCCA (Deep Generalized CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.aclweb.org/anthology/W19-4301.pdf
An alternative objective based on the linear GCCA solution. Can be extended to more than 2 views

DMCCA (Deep Multiset CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~

https://arxiv.org/abs/1904.01775
An alternative objective based on the linear MCCA solution. Can be extended to more than 2 views

DCCAE (Deep Canonically Correlated Autoencoders):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://proceedings.mlr.press/v37/wangb15.pdf

DVCCA/DVCCA Private (Deep variational CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://arxiv.org/pdf/1610.03454.pdf

Installation
------------

Install $cca-zoo by running:

pip install cca-zoo

Contribute
----------

- Issue Tracker: github.com/jameschapman19/cca_zoo/issues
- Source Code: github.com/jameschapman19/cca_zoo

Support
-------

If you are having issues, please let me know. This is my first python package so I am open to all and any feedback!
james.chapman.19@ucl.ac.uk

License
-------

The project is licensed under the MIT license.