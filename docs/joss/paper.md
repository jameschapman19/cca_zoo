---
title: 'cca-zoo: a python package for implementing regularized, deep and probabilistic variants of canonical correlation
analysis and partial least squares models.' tags:

- Python
- Multiview
- Machine Learning authors:
- name: James Chapman affiliation: "1" # (Multiple affiliations must be quoted)
- name: Hao-Ting Wang affiliation: 2 affiliations:
- name: Centre for Medical Image Computing, University College London, London, UK index: 1
- name: Centre for Medical Image Computing, University College London, London, UK index: 1 date: 19/08/2021 date: 31
  August 2021 bibliography: paper.bib
---

# Summary

It is increasingly common to collect large datasets with multiple views. Examples include natural language processing,
neuroimaging, multiomics and audiovisual data. Canonical Correlation Analyis (CCA) `@Hotelling:2001`  and Partial Least
Squares `@Hotelling:2001` are classical methods for investigating and quantifying multivariate relationships between
these views of data. The goal of CCA and its variants is to find projections (and associated weights) for each view of
the data into a latent space where they are highly correlated.

The original CCA has been developed into a family of models which include regularised `@Vinod:2001`,
kernelized `@ShaweTaylor:2001`, probabilistic/generative `@Jordan:2001`, and deep learning based `@Andrew:2001`
variants. In particular these have allowed practioners to apply these models to complex, high dimensional data.
Similarly, variants of PLS have been proposed including the widely used Penalized Matrix Decomposition
algorithm `@Witten:2001` which induces sparsity in the weight vectors.

`cca-zoo` is a Python package that implements a number of these variants in a simple API with standardised outputs.
While there are a few Python packages containing implementations of CCA, we would like to highlight the unique benefits
our package brings to the community. Firstly, `cca-zoo` contains a number of models for implementing various forms of
regularised CCA and PLS that have previously only been available using MATLAB and R. We hope that this will give Python
users access to these powerful models for both application and the development of new algorithms. Secondly, `cca-zoo`
contains implementations of a number of deep CCA variants written in PyTorch
`@PyTorch:2001` which are written in a modular style, allowing users to swap out neural network architectures for each
view and plug the models into a customised training pipeline. We also provide a standard pipeline which can serve as a
simple baseline. Thirdly, `cca-zoo` contains reference implementations of variational and deep variational CCA in
python. Finally, `cca-zoo` gives users the ability to simulate data containing specified correlation structures as well
as the paired MNIST data commonly used as a toy dataset in deep multiview learning.

# Statement of need

Much of the methods research in CCA and PLS methods has been coded in R. Despite this, the python ecosystem for
multiview learning now provides a few options. `scikit-learn` contains standard implementations of both CCA and PLS for
two-view data which plug into their mature API. `pyrcca` contains implementations of ridge regularised and kernelized
two-view CCA. The embed module of `mvlearn` is perhaps the closest relative of our package, containing implementations
of ridge regularised and kernelized multi-view CCA. They also implement a reference deep CCA using pytorch but its lack
of modularity makes it somewhat unflexible for application or development.

`cca-zoo` extends the existing ecosystem by providing implementations of a number of models for sparse regularised CCA
which have found popularity in genetics and neuroimaging where signals are contained in a small subset of variables.
With applications like these in mind, we also have tried to make it as simple as possible to access the learnt model
weights to perform further analysis in the feature space. Furthermore, `cca-zoo` contains modular implementations of
deep CCA as well as the only installable implementation of multiview deep CCA. Finally, `cca-zoo` adds generative models
including variational and deep variational CCA as well as higher order canonical correlation analysis with tensor and
deep tensor CCA.

# Implementation

`cca-zoo` models are built in a similar way to those in `scikit-learn`. The user first instantiates a model object and
its relevant hyperparameters. Next they call the model's fit() method. After fitting, the model object contains its
relevant parameters such as weights or dual coefficients (for kernel methods) which can be accessed for further
analysis. For iterative models, the model may also contain information about the convergence of the objective function.
After the model has been fit, its transform() method can be used to project views into latent variables and score() can
be used to measure the canonical correlations.

# Conclusion

`cca-zoo`

We hope that `cca-zoo` will help researchers to apply and develop Canonical Correlation Analysis and Partial Least
Squares models and continue to welcome contributions from the community.

# Citations

# Figures

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

JC is supported by the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (
i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College
London Hospitals.

# References