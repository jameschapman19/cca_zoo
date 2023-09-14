CCA-Zoo: Dive into the World of Canonical Correlation Analysis
==============================================================

Welcome to CCA-Zoo, your one-stop-shop for Canonical Correlation Analysis (CCA) algorithms. 
Over the last couple of decades a 'zoo' of different extensions of CCA have been proposed in academia.
We leverage modern computational methods, to make it quick and easy to apply these algorithms to real-world multi-view data.

Why CCA-Zoo?
------------

1. **Comprehensive**: From traditional linear CCA to deep learning versions like Deep CCA, CCA-Zoo offers an extensive range of algorithms.
2. **User-Friendly**: With an intuitive API design, even newcomers can swiftly familiarize themselves with CCA concepts.
3. **Performance**: Leveraging the power of Python and efficient computational libraries, CCA-Zoo ensures robust and speedy analysis.
4. **Active Community**: Being open-source, CCA-Zoo thrives on community contributions, making it ever-evolving and up-to-date.

What is CCA?
------------

Canonical Correlation Analysis (CCA) is a technique used to understand the structure of correlations between two or more views of data.
CCA is usually defined by finding transformations to apply to each view of the data such that the (lower dimensional) transformed variables are maximally correlated.
However, CCA has many other equivalent formulations, e.g. as a latent variable model, which may be more appropriate depending on your situation.

There are two main ways that classical CCA is used in practice.
Either one is interested in the lower dimensional transformed variables and uses them as inputs to a downstream task (such as classification, clustering, or non-linear dimension reduction); alternatively, one may be more intereted in interpreting the transformation themselves to better understand the joint distribution of the data.
We try to provide a comprehensive framework making all such interpretations straightforward.

We mentioned that there were many modern extensions of CCA; notably to regularised, non-linear, and probabilistic settings.
Generally these extensions are each derived from a particular formulation of CCA, and it becomes important which particular formulation is used for interpretation.
We try to include brief descriptions of where these extensions come from and how they are best used in practice.


Definitions and Conventions
---------------------------
Classical Population CCA
^^^^^^^^^^^^^^^^^^^^^^^^

First, we formalize the (classical, two-view) *population* CCA problem, using careful mathematical notation.

Suppose we have two random variables, :math:`X,Y` taking values in :math:`\mathbb{R}^p,\mathbb{R}^q` respectively with joint covariance partitioned as:

.. math::

    \Sigma = \left(\begin{array}{cc}
        \Sigma_{xx} & \Sigma_{x y} \\
        \Sigma_{y x} & \Sigma_{yy}
    \end{array}\right)

CCA finds successive vectors :math:`\mathbf{u}_k \in \mathbb{R}^p, \mathbf{v}_k \in \mathbb{R}^q` for :math:`k=1,\dots,K \leq \min(p,q)` solving:

.. math::

    \begin{align*}
        \underset{u \in \mathbb{R}^{p}, v \in \mathbb{R}^{q}}{\operatorname{maximize}}\, u^\top \Sigma_{x y} v \\
        \text{subject to  }& u^\top \Sigma_{xx} u \leq 1,\: v^\top \Sigma_{yy} v \leq 1,\\ 
        &u^\top \Sigma_{xx} u_j= v^\top \Sigma_{yy} v_j =0 \text{ for } 1 \leq j \leq k-1 .    
    \end{align*}

In particular, the :math:`k^\text{th}` pair maximizes :math:`\operatorname{Cor}(u_k^\top X, v_k^\top Y)` subject to the orthogonality constraints.

We call the optimal value :math:`\rho_k` the :math:`k^\text{th}` *canonical correlation*, call :math:`u_k,v_k` the :math:`k^\text{th}` pair of *canonical directions*, or simply *weights*, and call the transformed variables :math:`u_k^\top X, v_k^\top Y` the first pair of *canonical variates*. The quantities :math:`\Sigma_{xx} u_k,\Sigma_{yy} v_k` will also be important; we will refer to these as *canonical loading vectors*, or simply *loadings*.

.. math::

    \Sigma_{xx} u_k = (\operatorname{Cov}(X_j, X^T u_k))_{j=1}^p, \quad \Sigma_{yy} v_k = (\operatorname{Cov}(Y_j, Y^T v_k))_{j=1}^q

We also define the :math:`k^th` set of structure correlations as:

.. math::

    (\operatorname{Corr}(X_j, X^T u_k))_{j=1}^p, \quad \Sigma_{yy} v_k = (\operatorname{Corr}(Y_j, Y^T v_k))_{j=1}^q

Some notes:

- The canonical correlations are unique, but the weights and loadings are not unique in general.
- We can extend the :math:`(u_k)_k,(v_k)_k` to bases of :math:`\mathbb{R}^p,\mathbb{R}^q` respectively.
- For clarity, we shall always refer to the components of the original :math:`X,Y` as *variables*, and to the transformed variables as (canonical) *variates*.
