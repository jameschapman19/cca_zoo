Mathematical Foundations
===========================

Canonical Correlation Analysis (CCA) and Partial Least Squares (PLS) models
are effective ways of finding associations between multiple views of data.

PCA
----

It is helpful to start off by formulating PCA in its mathematical form.
The first principle component can be written as the solution to the convex optimisation problem:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_1w_1  \}

    \text{subject to:}

    w_1^Tw_1=1

WHich is optimized for the singular vectors of the covariance matrix :math:`X^TX`.

PLS
----

Now consider two data matrices with the same number of samples :math:`X_1` and :math:`X_2`.
It is tempting to write a slightly different optimisation problem:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}

    \text{subject to:}

    w_1^Tw_1=1

    w_2^Tw_2=1

Which is optimised for the left and right singular vectors of the cross covariance matrix :math:`X_1^TX_2`


CCA
----

To arrive at Canonical Correlation we change the constraints slightly so that they now depend on the variance
in each view as well as the weights. This makes them data dependent.

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}

    \text{subject to:}

    w_1^TX_1^TX_1w_1=1

    w_2^TX_2^TX_2w_2=1

Despite these more complex constraints, CCA is the solution to a generalized eigenvalue problem.

Regularized CCA
-----------------

Notice that the constraints for PLS and CCA are identical in the case that :math:`X_i^TX_i=I`.
This leads naturally to mixing the constraints and in particular the PLS constraint acts in a similar
way to the 'ridge' regularisation in Ridge Regression.

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}

    \text{subject to:}

    (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=1

    (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=1

Other regularized CCA and PLS
--------------------------------

There has been lots of research into more general forms of regularisation for CCA, in particular forms that induce
sparsity on the weights. In general these can be written in a form somewhat similar to:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2 + \lambda_1f_1(w_1) + \lambda_2f_2(w_2)  \}

    \text{subject to:}

    (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=1

    (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=1

These problems usually have no closed form solution and are typically solved by alternately fixing :math:`w_1`
and :math:`w_2`.

Kernel CCA and PLS
---------------------

By expressing the weights as :math:`w_i=\alpha_iX_i` we can transform the CCA problem into its kernel form:

.. math::

    \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{ \alpha_1^TK_1^TK_2\alpha_2  \}

    \text{subject to:}

    \alpha_1^TK_1^TK_1\alpha_1=1

    \alpha_2^TK_2^TK_2\alpha_2=1

Finally we can also consider more general kernel matrices without loss of generalisation. A similar reparameterization
exists for PLS and therefore for ridge regularized CCA.

Multiset CCA
----------------

Ridge Regularized CCA can be generalized to find correlations between more than one view with the formulation:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} w_i^TX_i^TX_jw_j  \}\\

    \text{subject to:}

    (1-c_i)w_i^TX_i^TX_iw_i+c_iw_i^Tw_i=1

This form is often referred to as SUMCOR CCA because it optimizes for the pairwise sum of . This can be formulated as a
generalized eigenvalue problem and therefore solved efficiently.

Generalized CCA
-----------------

Ridge Regularized CCA can be generalized to find correlations between more than one view with the formulation:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_iw_i^TX_i^TT  \}\\

    \text{subject to:}

    T^TT=1

This form is often referred to as MAXVAR CCA since it finds an auxiliary vector :math:`T` with fixed unit norm that has
maximum sum of variance with each view. This can also be formulated as a generalized eigenvalue problem and
therefore solved efficiently.

Deep CCA
----------

The ideas behind CCA can be extended to a general form where instead of linear weights, we consider functions
:math:`f(X_i)`. Where these functions are parameterized by neural networks, we can consider a 'Deep' CCA:

.. math::

    w_{opt}=\underset{w}{\mathrm{argmax}}\{ f(X_1)^Tf(X_2)  \}

    \text{subject to:}

    f(X_1)^Tf(X_1)=1

    f(X_2)^Tf(X_2)=1