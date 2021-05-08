Standard install
=================

Modules included in the standard pip install


Models
--------------------

.. currentmodule:: cca_zoo.models

Base CCA
^^^^^^^^^

This is used as the base for all the models in this package. By inheriting this class, other methods access transform,
fit_transform, and predict_corr and only differ in their fit methods (and transform where necessary).

.. autoclass:: _CCA_Base
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

CCA
^^^^^^^

.. autoclass:: CCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params
rCCA
^^^^^^^

.. autoclass:: rCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

MCCA
^^^^^^^

.. autoclass:: MCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

KCCA
^^^^^^^

.. autoclass:: KCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

GCCA
^^^^^^^

.. autoclass:: GCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

TCCA
^^^^^^^^^^^^^^^

.. autoclass:: TCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params


KTCCA
^^^^^^^^^^^^^^^

.. autoclass:: KTCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

PLS
^^^^^^^

.. autoclass:: PLS
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

CCA_ALS
^^^^^^^

.. autoclass:: CCA_ALS
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

PMD
^^^^^^^

.. autoclass:: PMD
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

SCCA
^^^^^^^

.. autoclass:: SCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

PMD
^^^^^^^

.. autoclass:: PMD
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

ParkhomenkoCCA
^^^^^^^^^^^^^^^

.. autoclass:: ParkhomenkoCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

ElasticCCA
^^^^^^^^^^^^^^^

.. autoclass:: ElasticCCA
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

Inner Loops for iterative optimization
---------------------------------------

.. automodule:: innerloop
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params



Simulated Data and Toy Datasets
--------------------------------

.. automodule:: cca_zoo.data
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params

Plotting Utilities
----------------------------

.. automodule:: cca_zoo.plot_utils
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: get_params, set_params