Models
=======================

Base Class
--------------------------------

.. automodule:: cca_zoo.models.cca_base
    :members:
    :private-members: _CCA_Base
    :exclude-members: get_params, set_params

rCCA
---------------------------

.. automodule:: cca_zoo.models.rcca
    :inherited-members:
    :exclude-members: get_params, set_params

GCCA and KGCCA
---------------------------

.. automodule:: cca_zoo.models.gcca
    :inherited-members:
    :exclude-members: get_params, set_params

MCCA and KCCA
---------------------------

.. automodule:: cca_zoo.models.mcca
    :inherited-members:
    :exclude-members: get_params, set_params

Tensor Canonical Correlation Analysis
----------------------------------------

Tensor Canonical Correlation Analysis
**************************************
.. autoclass:: cca_zoo.models.tcca.TCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Kernel Tensor Canonical Correlation Analysis
**********************************************
.. autoclass:: cca_zoo.models.tcca.KTCCA
    :inherited-members:
    :exclude-members: get_params, set_params

More Complex Regularisation using Iterative Models
-----------------------------------------------------

.. toctree::
   :maxdepth: 4
   :caption: More Complex Regularisation using Iterative Models

   iterative.rst


