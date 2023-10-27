Introduction to CCA
=======================

Welcome to CCA-Zoo, a one-stop repository for state-of-the-art Canonical Correlation Analysis (CCA) algorithms. Over the years, a multitude of CCA extensions have emerged in the academic sphere. CCA-Zoo harnesses the power of modern computational techniques to facilitate the effortless application of these algorithms to real-world, multiview data scenarios.

Why Choose CCA-Zoo?
-------------------

1. **Comprehensive**: Spanning traditional linear CCA to neural network-based implementations like Deep CCA, we provide an exhaustive selection of algorithms.
2. **User-Friendly**: Featuring an intuitive API, CCA-Zoo empowers both newcomers and experts to quickly grasp and apply CCA algorithms.
3. **High Performance**: Built on Python and optimized through efficient computational libraries, CCA-Zoo guarantees robust, high-speed analyses.
4. **Vibrant Community**: As an open-source project, CCA-Zoo continually evolves through active community participation and contributions.

What is Canonical Correlation Analysis (CCA)?
----------------------------------------------

Canonical Correlation Analysis (CCA) is a versatile technique used for identifying correlated patterns across multiple views of data. Although traditionally designed to optimize correlation in reduced-dimensionality spaces, numerous equivalent formulations exist—such as latent variable models—that may suit specific needs.

CCA finds widespread application in two main ways: as a feature extraction technique for downstream tasks like classification or clustering, and as a tool for understanding the underlying joint distribution of multiview data. CCA-Zoo aims to provide a comprehensive framework that accommodates both these aspects.

Modern Extensions of CCA
------------------------

CCA has evolved over time, giving rise to variants such as regularized, non-linear, and probabilistic CCA. These extensions often originate from unique formulations of the classical algorithm and are best understood and applied in context. Brief guidelines on the origins and practical uses of these extensions are provided in this documentation.

Formal Definitions and Mathematical Framework
---------------------------------------------

Classical Population CCA
^^^^^^^^^^^^^^^^^^^^^^^^

Here, we formalize the classical two-view population CCA problem using rigorous mathematical notation.

Let :math:`X` and :math:`Y` be two random variables in :math:`\mathbb{R}^p` and :math:`\mathbb{R}^q` respectively, with a joint covariance matrix partitioned as follows:

.. math::

    \Sigma = \begin{pmatrix}
        \Sigma_{xx} & \Sigma_{xy} \\
        \Sigma_{yx} & \Sigma_{yy}
    \end{pmatrix}

The objective of CCA is to find vectors :math:`\mathbf{u}_k \in \mathbb{R}^p, \mathbf{v}_k \in \mathbb{R}^q` that maximize the canonical correlation, subject to specific orthogonality constraints. The mathematical formulation is as follows:

.. math::

    \begin{align*}
        \max_{u \in \mathbb{R}^{p}, v \in \mathbb{R}^{q}}\, u^\top \Sigma_{xy} v \\
        \text{s.t. }& u^\top \Sigma_{xx} u \leq 1,\: v^\top \Sigma_{yy} v \leq 1, \\
        &u^\top \Sigma_{xx} u_j= v^\top \Sigma_{yy} v_j =0, \text{ for } 1 \leq j \leq k-1 .
    \end{align*}

For additional details, including the interpretation of canonical directions, correlations, and variates, please refer to the subsequent sections of this documentation.

Note:
-----
- Canonical correlations are unique, but the associated weights and loadings are generally not.
- The terms *variables* and *variates* will be consistently used to differentiate between components of the original and transformed data.

