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
    affiliation: 2 
affiliations:
  - name: Centre for Medical Image Computing, University College London, London, UK 
    index: 1
  - name: Centre de Recherche de l'Institut Universitaire de Gériatrie de Montréal, Université de Montréal, Montréal, QC, Canada
    index: 2 
date: 1 September 2021 
bibliography: paper.bib
---

# Summary

Many scientific datasets contain multiple views of data. Examples include different languages in natural language
processing, as well as neuroimaging, multiomics and audiovisual data. Canonical Correlation Analyis (
CCA) [@hotelling1992relations]  and Partial Least Squares are classical methods for investigating and quantifying
multivariate relationships between these views of data. The goal of CCA and its variants is to find projections (and
associated weights) for each view of the data into a latent space where they are highly correlated.

The original CCA has been developed into a family of models which include regularised [@vinod1976canonical],
kernelized [@hardoon2004canonical], probabilistic/generative [@bach2005probabilistic], and deep learning
based [@andrew2013deep] variants. In particular these have allowed practitioners to apply these models to complex, high
dimensional data. Similarly, variants of PLS have been proposed including the widely used Penalized Matrix Decomposition
algorithm [@witten2009penalized] which induces sparsity in the weight vectors for interpretability and generalisation.

`cca-zoo` is a Python package that implements a number of these variants in a simple API with standardised outputs.
While there are a few Python packages containing implementations of CCA, we would like to highlight the unique benefits
our package brings to the community. Firstly, `cca-zoo` contains a number of models for implementing various forms of
regularised CCA and PLS suitable for high dimensional data that have previously only been available in installable
packages in R. We believe that this will give Python users access to these powerful models for both application and the
development of new algorithms. Secondly,
`cca-zoo` contains implementations of a number of deep CCA variants written in PyTorch
[@paszke2019pytorch] which are written in a modular style, allowing users to swap out neural network architectures for
each view and plug the models into their own training pipeline. Thirdly, `cca-zoo` contains generative models including
variational and deep variational CCA which can be used to model the multiview data generation process and even generate
new synthetic samples. Finally, `cca-zoo` gives users the ability to simulate data containing specified correlation
structures as well as the paired MNIST data commonly used as a toy dataset in deep multiview learning.

# Statement of need

The python ecosystem for multiview learning currently provides a few options for implementing CCA and PLS models.
`scikit-learn` [@pedregosa2011scikit] contains standard implementations of both CCA and PLS for two-view data which plug
into their mature API. `pyrcca` [@bilenko2016pyrcca] contains implementations of ridge regularised and kernelized
two-view CCA. The embed module of `mvlearn` [@perry2020mvlearn] is perhaps the closest relative of our package,
containing implementations of ridge regularised and kernelized multi-view CCA. They also implement a reference deep CCA
using pytorch but its lack of modularity makes it somewhat unflexible for application to new datadets or development.

Standard implementations of these state-of-the-art models help as benchmarks for methods development as well as easy
application to new datasets.`cca-zoo` extends the existing ecosystem by providing implementations of a number of models
for sparse regularised CCA which have found popularity in genetics and neuroimaging where signals are contained in a
small subset of variables. With applications like these in mind, we also have tried to make it as simple as possible to
access the learnt model weights to perform further analysis in the feature space. Furthermore, `cca-zoo` contains
modular implementations of deep CCA and its multiview variants which allow the user to focus on architecture tuning.
Finally, `cca-zoo` adds generative models including variational [@wang2007variational] and deep variational
CCA [@wang2016deep] as well as higher order canonical correlation analysis with tensor [@kim2007tensor] and deep tensor
CCA [@wong2021deep].

# Implementation

`cca-zoo` models are built in a similar way to those in `scikit-learn`. The user first instantiates a model object and
its relevant hyperparameters. Next they call the model's fit() method. After fitting, the model object contains its
relevant parameters such as weights or dual coefficients (for kernel methods) which can be accessed for further
analysis. For models fit with iterative algorithms, the model may also contain information about the convergence of the
objective function. After the model has been fit, its transform() method can be used to project views into latent
variables and score() can be used to measure the canonical correlations.

Due to the size of the PyTorch and NumPyro packages used respectively in the deep and probabilistic models, these models
are only installed when the optional [deep] and [probabilistic] are passed during pip install. The complete list of
models along with their installation tag are provided below.

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
Partial Least Squares models and we continue to welcome contributions from the community.

# Acknowledgements

JC is supported by the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (
i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College
London Hospitals.

# References