Models
=======================


Regularized Canonical Correlation Analysis and Partial Least Squares
------------------------------------------------------------------------

Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.CCA
    :inherited-members: BaseEstimator

Partial Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.PLS
    :inherited-members: BaseEstimator

Ridge Regularized Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._rcca.rCCA
    :inherited-members: BaseEstimator

GCCA and KGCCA
---------------------------

Generalized (MAXVAR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._gcca.GCCA
    :inherited-members: BaseEstimator

Kernel Generalized (MAXVAR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._gcca.KGCCA
    :inherited-members: BaseEstimator

MCCA and KCCA
---------------------------

Multiset (SUMCOR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._mcca.MCCA
    :inherited-members: BaseEstimator

Kernel Multiset (SUMCOR) Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._mcca.KCCA
    :inherited-members: BaseEstimator

Tensor Canonical Correlation Analysis
----------------------------------------

Tensor Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._tcca.TCCA
    :inherited-members: BaseEstimator

Kernel Tensor Canonical Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models._tcca.KTCCA
    :inherited-members: BaseEstimator

More Complex Regularisation using Iterative Models
-----------------------------------------------------

Normal CCA and PLS by alternating least squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quicker and more memory efficient for very large data


PLS by Alternating Least Squares
""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.PLS_ALS
    :inherited-members: BaseEstimator
    :show-inheritance:


Sparsity Inducing Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Penalized Matrix Decomposition (Sparse PLS)
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_PMD
    :inherited-members: BaseEstimator
    :show-inheritance:

Sparse CCA by iterative lasso regression
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_IPLS
    :inherited-members: BaseEstimator
    :show-inheritance:

Elastic CCA by MAXVAR
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ElasticCCA
    :inherited-members: BaseEstimator
    :show-inheritance:

Span CCA
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_Span
    :inherited-members: BaseEstimator

Parkhomenko (penalized) CCA
"""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ParkhomenkoCCA
    :inherited-members: BaseEstimator

Sparse CCA by ADMM
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_ADMM
    :inherited-members: BaseEstimator

Miscellaneous
-----------------------------------------------------

Nonparametric CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.NCCA
    :inherited-members: BaseEstimator

Partial CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.PartialCCA
    :inherited-members: BaseEstimator

Sparse Weighted CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cca_zoo.models.SWCCA
    :inherited-members: BaseEstimator

Base Class
--------------------------------

.. automodule:: cca_zoo.models._base
    :inherited-members: BaseEstimator
