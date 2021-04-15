Welcome to cca-zoo's documentation!
===================================


Documentation
=============

.. toctree::
   :maxdepth: 4
   :caption: Documentation:
   
   documentation/getting_started
   documentation/tutorials
   documentation/api

.. toctree::
   :maxdepth: 4
   :caption: Package Reference:

   package_reference/standard

.. toctree::
   :maxdepth: 4
   :caption: Developer Information

   developer_info/contribute
   developer_info/support
   developer_info/license
   
Features
--------

Linear CCA/PLS:
~~~~~~~~~~~~~~~

A variety of linear CCA and PLS methods implemented using alternating minimization methods for non-convex optimisation
based on the power method or alternating least squares.

GCCA (Generalized MAXVAR CCA):

The generalized eigenvalue problem form of generalized MAXVAR CCA. Maximises the squared correlation between each view projection and
a shared auxiliary vector of unit length.

https://academic.oup.com/biomet/article-abstract/58/3/433/233349?redirectedFrom=fulltext

MCCA (Multiset SUMCOR CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generalized eigenvalue problem form of multiset SUMCOR CCA. Maximises the pairwise sum of correlations between view
projections.

SCCA (Sparse CCA - Mai):
~~~~~~~~~~~~~~~~~~~~~~~~

A solution to the sparse CCA problem based on iterative rescaled lasso regression problems to ensure projections are unit length.

https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043?casa_token=pw8OSPmNkzEAAAAA:CcrMA_8g_2po011hQsGQXfiYyvtpBlSS6LJm-z_zANOg6t5YhpFZ-2YJNeCbJdHmT7GXIFZUU7gQl78

PMD (Sparse PLS/PMD/Penalized Matrix Decomposition - Witten):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solution to a sparse CCA problem based on penalized matrix decomposition. The relaxation and assumptions made make this method
more similar to an l1-regularized PLS

https://academic.oup.com/biostatistics/article/10/3/515/293026

PCCA (Penalized CCA - elastic net - Waaijenborg):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solution to the sparse CCA problem based on iterative rescaled elastic regression problems to ensure projections are unit length.

https://pubmed.ncbi.nlm.nih.gov/19689958/

SCCA_ADMM (Sparse canonical correlation analysis-Suo):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solution to the sparse CCA problem based on iterative rescaled lasso regression problems solved using ADMM.

https://arxiv.org/abs/1705.10865

Kernel CCA
~~~~~~~~~~

CCA solved using the kernel method. Adding regularisation in the linear case can be shown to be equivalent to regularised CCA.

Linear Kernel
RBF Kernel
Polynomial Kernels

DCCA (Deep CCA):
~~~~~~~~~~~~~~~~

Using either Andrew's original Tracenorm Objective or Wang's alternating least squares solution

https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
https://arxiv.org/pdf/1510.02054v1.pdf


DGCCA (Deep Generalized CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative objective based on the linear GCCA solution. Can be extended to more than 2 views

https://www.aclweb.org/anthology/W19-4301.pdf

DMCCA (Deep Multiset CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative objective based on the linear MCCA solution. Can be extended to more than 2 views

https://arxiv.org/abs/1904.01775

DCCAE (Deep Canonically Correlated Autoencoders):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://proceedings.mlr.press/v37/wangb15.pdf

DVCCA/DVCCA Private (Deep variational CCA):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://arxiv.org/pdf/1610.03454.pdf





