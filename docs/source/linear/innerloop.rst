.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Iterative Methods
=================

The CCA_Iterative class gives further flexibility to use iterative optimization methods.

In it's basic form, CCA_Iterative performs unregularized PLS but by inheriting this class
and changing its inner_loop argument.

.. autoclass:: cca_zoo.wrappers.CCA_Iterative
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Base Inner Loop
---------------
The base class defines the shared structure of each of the inner optimisation loops
(and itself by default optimises for PLS)

.. autoclass:: cca_zoo.innerloop.InnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

PLS
----

.. autoclass:: cca_zoo.innerloop.PLSInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

CCA
----

.. autoclass:: cca_zoo.innerloop.CCAInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Sparse CCA by Penalized Matrix Decomposition
--------------------------------------------

https://academic.oup.com/biostatistics/article/10/3/515/293026

.. autoclass:: cca_zoo.innerloop.PMDInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Sparse CCA by Penalization (Parkhomenko)
--------------------------------------------

.. autoclass:: cca_zoo.innerloop.ParkhomenkoInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Sparse CCA by Rescaled Lasso (Mai)
--------------------------------------------

https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043?casa_token=pw8OSPmNkzEAAAAA:CcrMA_8g_2po011hQsGQXfiYyvtpBlSS6LJm-z_zANOg6t5YhpFZ-2YJNeCbJdHmT7GXIFZUU7gQl78

.. autoclass:: cca_zoo.innerloop.SCCAInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Sparse CCA by ADMM (Suo)
--------------------------------------------

https://arxiv.org/abs/1705.10865

.. autoclass:: cca_zoo.innerloop.ADMMInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Elastic CCA by Rescaled Lasso (Waaijenborg)
--------------------------------------------

https://pubmed.ncbi.nlm.nih.gov/19689958/

.. autoclass:: cca_zoo.innerloop.ElasticInnerLoop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

