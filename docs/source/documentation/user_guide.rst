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

   linear_cca.fit(train_view_1, train_view_2)

Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^^^^

Some models require hyperparameters. We can either choose these manually when the model is instantiated or we can use the gridsearch_fit() method
to use a data driven approach.

.. sourcecode:: python

   from cca_zoo.models import rCCA

   #Candidates for regularisation in the first view
   c1 = [0.1, 0.3, 0.7, 0.9]
   #Candidates for regularisation in the second view
   c2 = [0.1, 0.3, 0.7, 0.9]
   #param_candidates expects a dictionary
   param_candidates = {'c': list(itertools.product(c1, c2))}

   #performs 5 fold cross validation using 2 parallel jobs, printing the results and producing a hyperparameter plot
   ridge = rCCA(latent_dims=latent_dims).gridsearch_fit(
        train_view_1,
        train_view_2,
        param_candidates=param_candidates,
        folds=5,
        verbose=True, jobs=2,
        plot=True)

Model Transforms
-----------------

One models are fit we can transform the data to latent projections for each view

.. sourcecode:: python
   projection_1,projection_2=ridge.transform(train_view_1,train_view_2)

In a similar way to scikit-learn we can also call fit_transform to complete both of these steps in one go:

.. sourcecode:: python
   projection_1,projection_2=ridge.fit_transform(train_view_1,train_view_2)

Model Evaluation
-----------------

We can evaluate models by their correlation in the latent space

.. sourcecode:: python
   correlation=ridge.score(train_view_1,train_view_2)

Model Weights
-----------------

In applications of cca, we are often interested in the model weights. These can be easily accessed as arrays with
#features x #latent_dimensions for each view.

.. sourcecode:: python
   view_1_weights=ridge.weights[0]
   view_2_weights=ridge.weights[1]


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

This produces a PyTorch.nn.Module object which can be updated in a customised training loop. As a quick start, we also
provide a DeepWrapper class which wraps the deep cca model and its training loop so that it shares the fit(), transform()
and score() methods of the other models in the package.

.. sourcecode:: python
   from cca_zoo.deepmodels import DeepWrapper
   dcca_model = DeepWrapper(dcca_model)
   #datasets can be pytorch datasets which output ((view_1,view_2),label) or 2 or more numpy arrays
   dcca_model.fit(train_dataset, val_dataset=val_dataset, epochs=epochs)

We can now use:

.. sourcecode:: python
   dcca_model.score(train_dataset)

And:

.. sourcecode:: python
   projection_1,projection_2=dcca_model.transform(train_dataset)