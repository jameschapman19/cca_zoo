
**User Guide**
===============

Discover how `cca-zoo` empowers your exploration of multiview data using Canonical Correlation Analysis (CCA) and its various innovative forms.

Model Fitting
-------------

**Preparing Your Data**

Firstly, ensure your data is appropriately preprocessed. In this example, we generate two views of synthetic data, each comprising 10 features:

.. sourcecode:: python

    import numpy as np

    # Create synthetic data with two views
    train_view_1 = np.random.normal(size=(100, 10))
    train_view_2 = np.random.normal(size=(100, 10))

    # Remove the mean to normalize data
    train_view_1 -= train_view_1.mean(axis=0)
    train_view_2 -= train_view_2.mean(axis=0)

**Initiating and Fitting Your Model**

To start, instantiate your CCA model, specifying the desired latent dimensions:

.. sourcecode:: python

    from cca_zoo.models import CCA

    latent_dims = 3
    linear_cca = CCA(latent_dims=latent_dims)

    # Fit the model
    linear_cca.fit([train_view_1, train_view_2])

Hyperparameter Tuning
---------------------

**Manual vs Data-Driven Approaches**

Hyperparameters can be manually set during model instantiation. Alternatively, the `gridsearch_fit()` method offers a systematic, data-driven tuning approach.

Consider the following example for the regularized CCA (rCCA):

.. sourcecode:: python

   from cca_zoo.models import rCCA
   from cca_zoo.model_selection import GridSearchCV

   # Custom scoring function returning mean correlation in latent space
   def scorer(estimator, X):
      dim_corrs = estimator.score(X)
      return dim_corrs.mean()

   # Define grid of potential regularization parameters for each view
   c1 = [0.1, 0.3, 0.7, 0.9]
   c2 = [0.1, 0.3, 0.7, 0.9]
   param_grid = {'c': [c1, c2]}

   cv = 5  # Specify number of folds for cross-validation

   # Grid search with rCCA
   ridge = GridSearchCV(rCCA(latent_dims=latent_dims), param_grid=param_grid,
                        cv=cv, verbose=True, scoring=scorer).fit([train_view_1, train_view_2]).best_estimator_

Model Transformations
----------------------

Following model fitting, transform your data to obtain latent projections for each view:

.. sourcecode:: python

   projections = ridge.transform([train_view_1, train_view_2])

Or employ the `fit_transform` for a simultaneous fit and transformation:

.. sourcecode:: python

   projections = ridge.fit_transform([train_view_1, train_view_2])

Model Evaluation
----------------

Evaluate your model by determining its correlation in the latent space:

.. sourcecode:: python

   correlation = ridge.score([train_view_1, train_view_2])

For tensor CCA models, this represents higher-order correlations within each dimension.

Extracting Model Weights
------------------------

For specific CCA applications, accessing model weights—i.e., the linear transformations mapping each view to the latent space—is crucial. Here's how:

.. sourcecode:: python

   view_1_weights = ridge.weights[0]
   view_2_weights = ridge.weights[1]

Unraveling Deep Models
----------------------

Deep models in `cca-zoo` harness neural networks as view encoders, offering a way to capture intricate relationships between views.

**Constructing Encoder Architectures**

Define encoder networks' architectures, like the following multi-layer perceptrons (MLPs) example:

.. sourcecode:: python

   from cca_zoo.deepmodels import architectures

   encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
   encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)

**Deep CCA Model Initiation**

Instantiate a deep CCA model using the encoders:

.. sourcecode:: python

   from cca_zoo.deepmodels import DCCA

   dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

The output is a PyTorch.nn.Module object, which can undergo updates in a custom training loop. Furthermore, the provided LightningModule class (from pytorch-lightning) simplifies the training of these models.