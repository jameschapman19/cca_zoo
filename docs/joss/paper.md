---
title: 'cca-zoo: a python package for implementing models from the canonical correlation analysis family'
tags:
  - Python
  - Multiview
  - Machine Learning 
authors:
  - name: James Chapman 
    orcid: 0000-0002-9364-8118 
    affiliation: 1
  - name: Hao-Ting Wang 
    orcid: 0000-0003-4078-2038 
    affiliation: "2, 3" 
affiliations:
  - name: Centre for Medical Image Computing, University College London, London, UK 
    index: 1
  - name: Centre de Recherche de l'Institut Universitaire de Gériatrie de Montréal, Université de Montréal, Montréal, QC, Canada
    index: 2 
  - name: Centre de Recherche de l'Hôpital du Sacré Coeur de Montréal, Université de Montréal, Montréal, QC, Canada
    index: 3 
date: 1 September 2021 
bibliography: paper.bib
---

# Summary

Multi-view data has gained visibility in scientific research. Examples include different languages in natural language 
processing, as well as neuroimaging, multiomics and audiovisual data. Canonical Correlation Analysis (CCA)
[@hotelling1992relations]  and Partial Least Squares (PLS) are classical methods for investigating and quantifying
multivariate relationships between these views of data. The goal of CCA and its variants is to find projections (and
associated weights) for each view of the data into a latent space where they are highly correlated.

The original CCA is constrained by the sample-to-feature ratio. The algorithm cannot produce a solution when the number 
of features in one view exceeds the number of samples. To overcome this restriction, the original CCA has been developed 
into a family of models which include regularised [@vinod1976canonical], kernelized [@hardoon2004canonical], 
probabilistic/generative [@bach2005probabilistic], and deep learning based [@andrew2013deep] variants. In particular 
these variations have allowed practitioners to apply these models to complex, high dimensional data. Similarly, 
variants of PLS have been proposed including the widely used Penalized Matrix Decomposition algorithm 
[@witten2009penalized] which induces sparsity in the weight vectors for interpretability and generalisation.

`cca-zoo` is a Python package that implements many variants in a simple API with standardised outputs. We would like to 
highlight the unique benefits our package brings to the community in comparison to other established Python packages 
containing implementations of CCA. Firstly, `cca-zoo` contains a number of regularised CCA and PLS for high dimensional 
data that have previously only been available in installable packages in R. Native Python implementation will give 
Python users convenient access to these powerful models for both application and the development of new algorithms. 
Secondly,`cca-zoo` contains several deep CCA variants written in PyTorch [@paszke2019pytorch]. We adopted a modular 
style allowing users to apply their desired neural network architectures for each view for their own training pipeline. 
Thirdly, `cca-zoo` contains generative models including variational and deep variational CCA. This class of variations 
can be used to model the multiview data generation process and even generate new synthetic samples. Finally, `cca-zoo` 
provides data simulation utilites to synthesize data containing specified correlation structures as well as the paired 
MNIST data commonly used as a toy dataset in deep multiview learning.

# Statement of need

The python ecosystem for multiview learning currently provides a few options for implementing CCA and PLS models.
`scikit-learn` [@pedregosa2011scikit] contains standard implementations of both CCA and PLS for two-view data which plug
into their mature API. `pyrcca` [@bilenko2016pyrcca] contains implementations of ridge regularised and kernelized
two-view CCA. The embed module of `mvlearn` [@perry2020mvlearn] is perhaps the closest relative of `cca-zoo`, containing
implementations of ridge regularised and kernelized multi-view CCA. `cca-zoo` builds on the `mvlearn` API by providing
an additional range of regularised models and in particular sparsity inducing models which have found success in
multiomics. Building on the reference implementation in `mvlearn`, `cca-zoo` further provides a number of deep learning
models with a modular design to enable users to supply their own choice of neural network architectures.

Standard implementations of state-of-the-art models help as benchmarks for methods development and easy application to
new datasets. `cca-zoo` extends the existing ecosystem with a number of sparse regularised CCA models. These variations
have found popularity in genetics and neuroimaging where signals are contained in a small subset of variables. With
applications like these in mind, `cca-zoo` simplified the access to the learnt model weights to perform further analysis
in the feature space. Furthermore, the modular implementations of deep CCA and its multiview variants allow the user to
focus on architecture tuning. Finally, `cca-zoo` adds generative models including variational [@wang2007variational] and
deep variational CCA [@wang2016deep] as well as higher order canonical correlation analysis with tensor [@kim2007tensor]
and deep tensor CCA [@wong2021deep].

# Implementation

`cca-zoo` adopted a similar API to that used in `scikit-learn`. The user first instantiates a model object and its
relevant hyperparameters. Next they call the model's `fit()` method to apply the data. After fitting, the model object
contains its relevant parameters such as weights or dual coefficients (for kernel methods) which can be accessed for
further analysis. For models that fit with iterative algorithms, the model may also contain information about the
convergence of the objective function. After the model has been fit, its `transform()` method can project views into
latent variables and `score()` can be used to measure the canonical correlations.

The deep and probabilistic models are supported by PyTorch and NumPyro respectively. Due to the size of these
dependencies, these two classes of variations are not in the default installation. Instead, we provide options [deep]
and [probabilistic] for users. The list bellow provides the complete collection of models along with their installation
tag is provided below.

## Model List

A complete model list at the time of publication:

| Model Class | Model Name | Number of Views | Install |
| -------- | -------- | ------ |-----|
| CCA   | Canonical Correlation Analysis | 2   | standard |
| rCCA   | Canonical Ridge | 2   | standard |
| KCCA   | Kernel Canonical Correlation Analysis | 2   | standard |
| MCCA   | Multiset Canonical Correlation Analysis | \>=2   | standard |
| KMCCA   | Kernel Multiset Canonical Correlation Analysis | \>=2   | standard |
| GCCA   | Generalized Canonical Correlation Analysis | \>=2   | standard |
| KGCCA   | Kernel Generalized Canonical Correlation Analysis | \>=2   | standard |
| PLS   | Partial Least Squares | \>=2   | standard |
| CCA_ALS   | Canonical Correlation Analysis by Alternating Least Squares) [@golub1995canonical] | \>=2   | standard |
| PLS_ALS   | Partial Least Squares by Alternating Least Squares)  | \>=2   | standard |
| PMD   | Sparse CCA by Penalized Matrix Decomposition | \>=2   | standard |
| ElasticCCA   | Sparse Penalized CCA [@waaijenborg2008quantifying] | \>=2   | standard |
| ParkhomenkoCCA   | Sparse CCA [@parkhomenko2009sparse] | \>=2   | standard |
| SCCA   | Sparse Canonical Correlation Analysis by Iterative Least Squares [@mai2019iterative] | \>=2   | standard |
| SCCA_ADMM   | Sparse Canonical Correlation Analysis by Altnerating Direction Method of Multipliers [@suo2017sparse] | \>=2   | standard |
| SpanCCA   | Sparse Diagonal Canonical Correlation Analysis [@asteris2016simple] | \>=2   | standard |
| SWCCA   | Sparse Weighted Canonical Correlation Analysis [@wenwen2018sparse] | \>=2   | standard |
| TCCA   | Tensor Canonical Correlation Analysis | \>=2   | standard |
| KTCCA   | Kernel Tensor Canonical Correlation Analysis [@kim2007tensor] | \>=2   | standard |
| DCCA   | Deep Canonical Correlation Analysis | \>=2   | deep |
| DCCA_NOI   | Deep Canonical Correlation Analysis by Non-Linear Orthogonal Iterations [@wang2015stochastic] | \>=2   | deep |
| DCCAE   | Deep Canonically Correlated Autoencoders [@wang2015deep] | \>=2   | deep |
| DTCCA   | Deep Tensor Canonical Correlation Analysis | \>=2   | deep |
| SplitAE   | Split Autoencoders [@ngiam2011multimodal] | 2   | deep |
| DVCCA   | Deep Variational Canonical Correlation Analysis | \>=2   | deep |
| VCCA   | Variational Canonical Correlation Analysis | 2   | probabilistic |

## Documentation

The package is accompanied by documentation (https://cca-zoo.readthedocs.io/en/latest/index.html) and a number of
tutorial notebooks which serve as both guides to the package as well as educational resources for CCA and PLS methods.

# Conclusion

`cca-zoo` fills many of the gaps in the multiview learning ecosystem in Python, including a flexible API for
deep-learning based models, regularised models for high dimensional data (and in particular those that induce sparsity),
and generative models.`cca-zoo` will therefore help researchers to apply and develop Canonical Correlation Analysis and
Partial Least Squares models. We continue to welcome contributions from the community.

# Acknowledgements

JC is supported by the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (
i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College
London Hospitals. HTW is supported by funds from la Fondation Courtois awarded to Dr. Pierre Bellec. 

# References
