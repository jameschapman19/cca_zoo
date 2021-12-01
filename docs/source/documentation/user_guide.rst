User Guide
===========

Model Fit
----------

.. sourcecode:: python

   from cca_zoo.models import CCA
   from cca_zoo.data import generate_covariance_data
   # %%
   (train_view_1,train_view_2),(true_weights_1,true_weights_2)=generate_covariance_data(n=200,view_features=[10,10],latent_dims=1,correlation=1)


   linear_cca = CCA(latent_dims=latent_dims, max_iter=max_iter)

   linear_cca.fit([train_view_1, train_view_2])

Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^^^^

Some models require hyperparameters. We can either choose these manually when the model is instantiated or we can use the gridsearch_fit() method
to use a data driven approach.

.. sourcecode:: python

   from cca_zoo.models import rCCA
   from cca_zoo.model_selection import GridsearchCV

    def scorer(estimator,X):
      dim_corrs=estimator.score(X)
      return dim_corrs.mean()

    c1 = [0.1, 0.3, 0.7, 0.9]
    c2 = [0.1, 0.3, 0.7, 0.9]
    param_grid = {'c': [c1,c2]}

    ridge = GridSearchCV(rCCA(latent_dims=latent_dims),param_grid=param_grid,
        cv=cv,
        verbose=True,scoring=scorer).fit([train_view_1,train_view_2]).best_estimator_

Model Transforms
-----------------

Once models are fit we can transform the data to latent projections for each view

.. sourcecode:: python

   projection_1,projection_2=ridge.transform([train_view_1,train_view_2])

In a similar way to scikit-learn we can also call fit_transform to complete both of these steps in one go:

.. sourcecode:: python

   projection_1,projection_2=ridge.fit_transform([train_view_1,train_view_2])

Model Evaluation
-----------------

We can evaluate models by their correlation in the latent space

.. sourcecode:: python

   correlation=ridge.score([train_view_1,train_view_2])

For most models this gives us the average pairwise correlation in each latent dimension. For tensor cca models this
gives the higher order correlation in each dimension.

Model Weights
-----------------

In applications of cca, we are often interested in the model weights. These can be easily accessed as arrays with
#features x #latent_dimensions for each view.

.. sourcecode:: python

   view_1_weights=ridge.weights[0]
   view_2_weights=ridge.weights[1]

Model Loadings
-----------------

Similarly we can access the loadings for a given set of samples

.. sourcecode:: python

   view_1_loadings, view_2_loadings=ridge.get_loadings([train_view_1, train_view_2])


Deep Models
------------

Deep models have a slightly more involved process. We first need to choose the architectures for our encoder models

.. sourcecode:: python

   from cca_zoo.deepmodels import architectures
   encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
   encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)

We build our deep cca model using these encoders as inputs:

.. sourcecode:: python

   from cca_zoo.deepmodels import DCCA
   dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

This produces a PyTorch.nn.Module object which can be updated in a customised training loop. We also provide a LightningModule
class from pytorch-lightning which can be used to train any of these models.