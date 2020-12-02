.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cca-zoo's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

cca-zoo
========

cca-zoo is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data.

Look how easy it is to use:

.. sourcecode:: python

   import cca_zoo
   # train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims)
   linear_cca.fit(train_set_1, train_set_2)

Features
--------

- Linear CCA
- Regularised CCA (Witten, Parkhomenko, Waaijenborg, Mai)
- Kernel CCA (e.g. Hardoon)
- Deep CCA (Andrew)
- Deep Canonically Correlated Autoencoders (Wang)
- Deep Variational CCA
- Generalized versions of all of the above (i.e. more than 2 shared views)

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