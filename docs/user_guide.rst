.. _user_guide:

User Guide
==========

Explore the robust capabilities of `cca-zoo` in facilitating multiview data analysis through Canonical Correlation Analysis (CCA) and its advanced variations.

Model Fitting
-------------

Preparing Your Data
~~~~~~~~~~~~~~~~~~~

Ensure your data is appropriately preprocessed before analysis. In this example, we create two synthetic views, each containing 10 features.

.. code-block:: python

    import numpy as np

    # Create synthetic data for two views
    train_view_1 = np.random.normal(size=(100, 10))
    train_view_2 = np.random.normal(size=(100, 10))

    # Normalize the data by removing the mean
    train_view_1 -= train_view_1.mean(axis=0)
    train_view_2 -= train_view_2.mean(axis=0)

Initiating and Fitting Your Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin, instantiate the CCA model and specify the desired number of latent dimensions.

.. code-block:: python

    from cca_zoo.models import CCA

    latent_dimensions = 3
    linear_cca = CCA(latent_dimensions=latent_dimensions)

    # Fit the model
    linear_cca.fit((train_view_1, train_view_2))

Hyperparameter Tuning
---------------------

Manual vs Data-Driven Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters can either be manually configured during model initialization or tuned in a data-driven manner using the `gridsearch_fit()` method.

.. code-block:: python

    from cca_zoo.models import rCCA
    from cca_zoo.model_selection import GridSearchCV

    # Custom scoring function
    def scorer(estimator, X):
        dim_corrs = estimator.score(X)
        return dim_corrs.mean()

    # Define grid of potential regularization parameters
    c1 = [0.1, 0.3, 0.7, 0.9]
    c2 = [0.1, 0.3, 0.7, 0.9]
    param_grid = {'c': [c1, c2]}

    cv = 5  # Number of folds in cross-validation

    # Conduct grid search
    ridge = GridSearchCV(rCCA(latent_dimensions=latent_dimensions), param_grid=param_grid,
                         cv=cv, verbose=True, scoring=scorer).fit((train_view_1, train_view_2)).best_estimator_

Model Transformations
----------------------

Transform your data post-fitting to obtain latent projections for each view.

.. code-block:: python

    projections = ridge.transform((train_view_1, train_view_2))

Alternatively, use `fit_transform` for simultaneous fitting and transformation.

.. code-block:: python

    projections = ridge.fit_transform((train_view_1, train_view_2))

Model Evaluation
----------------

Assess the performance of your model by evaluating the correlations in the latent space.

.. code-block:: python

    correlation = ridge.score((train_view_1, train_view_2))

For tensor-based CCA models, this score represents higher-order correlations in each dimension.

Extracting Model Weights
------------------------

In specialized applications, it may be essential to access the model's linear transformations for each view.

.. code-block:: python

    view_1_weights = ridge.weights_[0]
    view_2_weights = ridge.weights_[1]

Deep Models in CCA-Zoo
----------------------

Deep models in `cca-zoo` utilize neural networks as view encoders, capturing complex relationships between different views.

Constructing Encoder Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we define encoder architectures using multi-layer perceptrons (MLPs).

.. code-block:: python

    from cca_zoo.deepmodels import architectures

    encoder_1 = architectures.Encoder(latent_dimensions=latent_dimensions, feature_size=784)
    encoder_2 = architectures.Encoder(latent_dimensions=latent_dimensions, feature_size=784)

Deep CCA Model Initiation
~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize a Deep CCA model using the encoder architectures.

.. code-block:: python

    from cca_zoo.deepmodels import DCCA

    dcca_model = DCCA(latent_dimensions=latent_dimensions, encoders=[encoder_1, encoder_2])

The resulting object is a PyTorch.nn.Module, allowing for further updates in a custom training loop.