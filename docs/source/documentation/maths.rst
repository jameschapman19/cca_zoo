Mathematical Foundations
===========================

Canonical Correlation Analysis (CCA) and Partial Least Squares (PLS) models
are effective ways of finding associations between multiple views of data.

PCA
----

It is helpful to start off by formulating PCA in its mathematical form.
The first principle component can be written as the solution to the convex optimisation problem:

.. math::

    \w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_1w_1  \}

    \text{subject to:}

    w_1^Tw_1=1

That is the singular vectors of the covariance matrix :math:`X^TX`

PLS
----

Now consider two data matrices with the same number of samples :math:`X_1` and :math:`X_2`.
It is tempting to write a slightly different optimisation problem:

.. math::

    \w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}

    \text{subject to:}

    w_1^Tw_1=1

    w_2^Tw_2=1

Which is optimised for the left and right singular vectors of the cross covariance matrix :math:`X_1^TX_2`


CCA
----

To arrive at Canonical Correlation

.. math::

    \w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}

    \text{subject to:}

    w_1^TX_1^TX_1w_1=1

    w_2^TX_2^TX_2w_2=1


Deep CCA
----------

To arrive

.. math::

    \w_{opt}=\underset{w}{\mathrm{argmax}}\{ f(X_1)^Tf(X_2)  \}

    \text{subject to:}

    f(X_1)^Tf(X_1)=1

    f(X_2)^Tf(X_2)=1