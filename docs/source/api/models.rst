Models
=======================


Regularized Canonical Correlation Analysis and Partial Least Squares
------------------------------------------------------------------------

Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.CCA
    :inherited-members:
    :exclude-members: get_params, set_params

Partial Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.PLS
    :inherited-members:
    :exclude-members: get_params, set_params

Ridge Regularized Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.rCCA
    :inherited-members:
    :exclude-members: get_params, set_params

GCCA and KGCCA
---------------------------

Generalized (MAXVAR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._gcca.GCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Kernel Generalized (MAXVAR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._gcca.KGCCA
    :inherited-members:
    :exclude-members: get_params, set_params

MCCA and KCCA
---------------------------

Multiset (SUMCOR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._mcca.MCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Kernel Multiset (SUMCOR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._mcca.KCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Tensor Canonical Correlation Analysis
----------------------------------------

Tensor Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._tcca.TCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Kernel Tensor Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._tcca.KTCCA
    :inherited-members:
    :exclude-members: get_params, set_params

More Complex Regularisation using Iterative Models
-----------------------------------------------------

Normal CCA and PLS by alternating least squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quicker and more memory efficient for very large data


PLS by Alternating Least Squares
""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.PLS_ALS
    :inherited-members:
    :exclude-members: get_params, set_params


Sparsity Inducing Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Penalized Matrix Decomposition (Sparse PLS)
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_PMD
    :inherited-members:
    :exclude-members: get_params, set_params

Sparse CCA by iterative lasso regression
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_IPLS
    :inherited-members:
    :exclude-members: get_params, set_params

Elastic CCA by MAXVAR
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ElasticCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Span CCA
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_Span
    :inherited-members:
    :exclude-members: get_params, set_params

Parkhomenko (penalized) CCA
"""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ParkhomenkoCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Sparse CCA by ADMM
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_ADMM
    :inherited-members:
    :exclude-members: get_params, set_params

Miscellaneous
-----------------------------------------------------

Nonparametric CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.NCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Partial CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.PartialCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Sparse Weighted CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.SWCCA
    :inherited-members:
    :exclude-members: get_params, set_params

Base Class
--------------------------------

.. automodule:: cca_zoo.models._base
    :members:
    :private-members: _BaseCCA
    :exclude-members: get_params, set_params
