User Guide
===========

This guide will show you how to use cca-zoo to perform canonical correlation analysis (CCA) and its variants on multiview data.

Model Fit
----------

To fit a CCA model, you need to provide two or more views of the same data as numpy arrays. For example, you can generate some synthetic data with two views and 10 features each:

.. sourcecode:: python

    import numpy as np

    train_view_1 = np.random.normal(size=(100, 10))
    train_view_2 = np.random.normal(size=(100, 10))
    # Remove mean
    train_view_1 -= train_view_1.mean(axis=0)
    train_view_2 -= train_view_2.mean(axis=0)

Then, you can import the CCA model and create an instance with the desired number of latent dimensions:

.. sourcecode:: python

    from cca_zoo.models import CCA

    latent_dims=3
    linear_cca = CCA(latent_dims=latent_dims)

Finally, you can fit the model to the data by passing a list of the views:

.. sourcecode:: python

    linear_cca.fit([train_view_1, train_view_2])

Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^^^^

Some models have hyperparameters that need to be tuned for optimal performance. You can either choose these manually when creating the model instance, or you can use the gridsearch_fit() method to use a data-driven approach.

For example, if you want to use regularized CCA (rCCA), you need to choose the regularization parameters for each view. You can define a grid of possible values and use the GridSearchCV class to find the best combination:

.. sourcecode:: python

   from cca_zoo.models import rCCA
   from cca_zoo.model_selection import GridSearchCV

   # Define a custom scoring function that returns the mean correlation in the latent space
   def scorer(estimator,X):
      dim_corrs=estimator.score(X)
      return dim_corrs.mean()

   # Define a grid of regularization parameters for each view
   c1 = [0.1, 0.3, 0.7, 0.9]
   c2 = [0.1, 0.3, 0.7, 0.9]
   param_grid = {'c': [c1,c2]}

   # Define the number of folds for cross-validation
   cv = 5

   # Create and fit a grid search object with rCCA as the base estimator
   ridge = GridSearchCV(rCCA(latent_dims=latent_dims),param_grid=param_grid,
        cv=cv,
        verbose=True,scoring=scorer).fit([train_view_1,train_view_2]).best_estimator_

Model Transforms
-----------------

Once models are fit, you can transform the data to latent projections for each view. This will return a list of numpy arrays with the same number of samples as the input data and as many columns as latent dimensions:

.. sourcecode:: python

   projection_1,projection_2=ridge.transform([train_view_1,train_view_2])

In a similar way to scikit-learn, you can also call fit_transform to complete both fitting and transforming steps in one go:

.. sourcecode:: python

   projection_1,projection_2=ridge.fit_transform([train_view_1,train_view_2])

Model Evaluation
-----------------

You can evaluate models by their correlation in the latent space. This will return a numpy array with as many elements as latent dimensions:

.. sourcecode:: python

   correlation=ridge.score([train_view_1,train_view_2])

For most models this gives us the average pairwise correlation in each latent dimension. For tensor CCA models this
gives the higher order correlation in each dimension.

Model Weights
-----------------

In some applications of CCA, we are interested in the model weights. These are the linear transformations that map each view to the latent space. You can easily access them as numpy arrays with #features x #latent_dimensions for each view:

.. sourcecode:: python

   view_1_weights=ridge.weights[0]
   view_2_weights=ridge.weights[1]


Deep Models
------------

Deep models are CCA models that use neural networks as encoders for each view. They allow us to capture nonlinear relationships between views and learn more expressive representations.

To use deep models, we first need to choose the architectures for our encoder networks. For example, we can use simple multilayer perceptrons (MLPs) with 784 input features and 3 latent dimensions:

.. sourcecode:: python

   from cca_zoo.deepmodels import architectures
   encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
   encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)

We can then build our deep CCA model using these encoders as inputs:

.. sourcecode:: python

   from cca_zoo.deepmodels import DCCA
   dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

This produces a PyTorch.nn.Module object which can be updated in a customised training loop. We also provide a LightningModule
class from pytorch-lightning which can be used to train any of these models with minimal boilerplate code.

