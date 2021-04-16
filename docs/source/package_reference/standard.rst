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
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

CCA
^^^^^^^

.. autoclass:: CCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

rCCA
^^^^^^^

.. autoclass:: rCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

PLS
^^^^^^^

.. autoclass:: PLS
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

MCCA
^^^^^^^

.. autoclass:: MCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

GCCA
^^^^^^^

.. autoclass:: GCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

TCCA
^^^^^^^^^^^^^^^

.. autoclass:: TCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

KCCA
^^^^^^^

.. autoclass:: KCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

CCA_ALS
^^^^^^^

.. autoclass:: CCA_ALS
    :exclude-members:

PMD
^^^^^^^

.. autoclass:: PMD
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

SCCA
^^^^^^^

.. autoclass:: SCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

PMD
^^^^^^^

.. autoclass:: PMD
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

ParkhomenkoCCA
^^^^^^^^^^^^^^^

.. autoclass:: ParkhomenkoCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

ElasticCCA
^^^^^^^^^^^^^^^

.. autoclass:: ElasticCCA
    :exclude-members:
    :show-inheritance:
    :special-members: __init__

Inner Loops for iterative optimization
---------------------------------------

.. automodule:: cca_zoo.innerloop
   :members:
   :undoc-members:
   :show-inheritance: _InnerLoop
   :private-members: _InnerLoop
   :special-members: __init__

Kernel CCA
----------------------------

.. automodule:: cca_zoo.kcca
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Simulated Data and Toy Datasets
--------------------------------

.. automodule:: cca_zoo.data
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Plotting Utilities
----------------------------

.. automodule:: cca_zoo.plot_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__