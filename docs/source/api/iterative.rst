Normal CCA and PLS by alternating least squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quicker and more memory efficient for very large data


CCA by Alternating Least Squares
""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.CCA_ALS
    :inherited^members:
    :exclude^members: get_params, set_params

PLS by Alternating Least Squares
""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.PLS_ALS
    :inherited^members:
    :exclude^members: get_params, set_params


Sparsity Inducing Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Penalized Matrix Decomposition (Sparse PLS)
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.PMD
    :inherited^members:
    :exclude^members: get_params, set_params

Sparse CCA by iterative lasso regression
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA
    :inherited^members:
    :exclude^members: get_params, set_params

Elastic CCA by MAXVAR
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ElasticCCA
    :inherited^members:
    :exclude^members: get_params, set_params

Span CCA
"""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SpanCCA
    :inherited^members:
    :exclude^members: get_params, set_params

Parkhomenko (penalized) CCA
"""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.ParkhomenkoCCA
    :inherited^members:
    :exclude^members: get_params, set_params

Sparse CCA by ADMM
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SCCA_ADMM
    :inherited^members:
    :exclude^members: get_params, set_params

Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sparse Weighted CCA
"""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: cca_zoo.models.SWCCA
    :inherited^members:
    :exclude^members: get_params, set_params