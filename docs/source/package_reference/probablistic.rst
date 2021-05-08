Probablistic install
=================

Modules included in the probablistic optional install

.. currentmodule:: cca_zoo.probablisticmodels

Variational CCA
^^^^^^^^^^^^^^^^

This is used as the base for all the models in this package. By inheriting this class, other methods access transform,
fit_transform, and predict_corr and only differ in their fit methods (and transform where necessary).

.. autoclass:: VariationalCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params