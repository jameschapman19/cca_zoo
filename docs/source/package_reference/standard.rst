Standard install
=================

Modules included in the standard pip install


Model Wrappers
--------------------

.. currentmodule:: cca_zoo.wrappers

Base CCA
^^^^^^^^^

This is used as the base for all the models in this package. By inheriting this class, other methods access transform,
fit_transform, and predict_corr and only differ in their fit methods (and transform where necessary).

.. autoclass:: _CCA_Base
   :members:
   :inherited-members:

CCA
^^^^^^^

.. autoclass:: CCA
   :members:
   :inherited-members:
   :show-inheritance:

rCCA
^^^^^^^

.. autoclass:: rCCA
   :members:
   :inherited-members:
   :show-inheritance:

PLS
^^^^^^^

.. autoclass:: PLS
   :members:
   :inherited-members:
   :show-inheritance:

MCCA
^^^^^^^

.. autoclass:: MCCA
   :members:
   :inherited-members:
   :show-inheritance:

GCCA
^^^^^^^

.. autoclass:: GCCA
   :members:
   :inherited-members:
   :show-inheritance:

TCCA
^^^^^^^^^^^^^^^

.. autoclass:: TCCA
   :members:
   :inherited-members:
   :show-inheritance:

KCCA
^^^^^^^

.. autoclass:: KCCA
   :members:
   :inherited-members:
   :show-inheritance:

CCA_ALS
^^^^^^^

.. autoclass:: CCA_ALS
   :members:
   :inherited-members:
   :show-inheritance:

PMD
^^^^^^^

.. autoclass:: PMD
   :members:
   :inherited-members:
   :show-inheritance:

SCCA
^^^^^^^

.. autoclass:: SCCA
   :members:
   :inherited-members:
   :show-inheritance:

PMD
^^^^^^^

.. autoclass:: PMD
   :members:
   :inherited-members:
   :show-inheritance:

ParkhomenkoCCA
^^^^^^^^^^^^^^^

.. autoclass:: ParkhomenkoCCA
   :members:
   :inherited-members:
   :show-inheritance:

ElasticCCA
^^^^^^^^^^^^^^^

.. autoclass:: ElasticCCA
   :members:
   :inherited-members:
   :show-inheritance:

Inner Loops for iterative optimization
---------------------------------------

.. automodule:: cca_zoo.innerloop
   :members:
   :inherited-members:
   :show-inheritance:

Kernel CCA
----------------------------

.. automodule:: cca_zoo.kcca
   :members:
   :inherited-members:
   :show-inheritance:

Simulated Data and Toy Datasets
--------------------------------

.. automodule:: cca_zoo.data
   :members:
   :inherited-members:
   :show-inheritance:

Plotting Utilities
----------------------------

.. automodule:: cca_zoo.plot_utils
   :members:
   :inherited-members:
   :show-inheritance: