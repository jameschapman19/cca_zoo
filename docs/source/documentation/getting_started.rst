Getting Started
===============

cca-zoo is a Python package for canonical correlation analysis (CCA) and its variants. CCA is a technique for finding the linear relationships between two or more views of the same data. cca-zoo provides a collection of linear, kernel, and deep methods for CCA of multiview data.

cca-zoo follows the scikit-learn/mvlearn APIs and models, therefore they have
fit/transform/fit_transform methods as standard.

Here is a simple example of how to use cca-zoo:

.. sourcecode:: python

    # Import the CCA model and the data generator
    from cca_zoo.models import CCA
    import numpy as np

    # Generate some data
    train_view_1 = np.random.normal(size=(100, 10))
    train_view_2 = np.random.normal(size=(100, 10))
    # Remove mean
    train_view_1 -= train_view_1.mean(axis=0)
    train_view_2 -= train_view_2.mean(axis=0)

    # Create and fit a linear CCA model
    linear_cca = CCA(latent_dims=latent_dims)
    linear_cca.fit((train_view_1, train_view_2))

    # Transform the data to the latent space
    train_view_1_latent, train_view_2_latent = linear_cca.transform((train_view_1, train_view_2))

