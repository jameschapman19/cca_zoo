.. currentmodule:: cca_zoo

.. _glossary:

=========================================
Glossary of Common Terms and API Elements
=========================================

This glossary hopes to definitively represent the tacit and explicit
conventions applied in CCA-Zoo and its API, while providing a reference
for users and contributors. It aims to describe the concepts and either detail
their corresponding API or link to other relevant parts of the documentation
which do so. By linking to glossary entries from the API Reference and User
Guide, we may minimize redundancy and inconsistency.

Where possible, we follow the conventions from scikit-learn's glossary.

Concepts Different from scikit-learn
=====================================

.. glossary::

    ``weights_``
        In the context of CCA-Zoo, ``weights_`` refers to the learned weights of a model, particularly for linear models. This attribute stores the weight vectors for each view in multivariate latent variable linear models. It differs from the ``coef_`` attribute commonly used in scikit-learn to denote model weights. For kernel-based models, ``weights_`` represents the dual coefficients, often denoted as ``alpha`` in the literature. Note that ``weights_`` is not applicable to non-linear models.

    ``loadings_``
        The ``loadings_`` attribute represents the normalized weights of the model. In CCA-Zoo models, these loadings are specifically tailored to reflect the correlation between the original variables in each view and their respective canonical variates. Loadings are normalized to ensure values range between -1 and 1, providing insights into the significance of original variables in defining the canonical variates.

    ``canonical_loadings_``
        This attribute calculates the canonical loadings for each view. Canonical loadings indicate the correlation between the original variables in a view and their respective canonical variates, which are linear combinations of the original variables formed to maximize correlation with variates from another view. These loadings offer a deeper understanding of how each original variable contributes to the model's extracted features.

    ``canonical_correlations_``
        The ``canonical_correlations_`` term refers to the pairwise correlations between the representations in each latent dimension. It is a measure of the strength of the relationship between the different sets of variables (views) in the model, calculated post-transformation.

    ``scores``
        In CCA-Zoo, ``scores`` generally refer to the output of the `score` method, which calculates the sum of average pairwise correlations between representations. It provides a singular value to assess the overall performance or quality of the model's fit to the data.

    ``explained_variance_``
        The ``explained_variance_`` attribute quantifies the variance captured by each latent dimension for each view in the model. It offers a view-specific understanding of the contribution of each latent dimension to the total variance, aiding in interpreting the model's effectiveness in capturing the data's variance.

