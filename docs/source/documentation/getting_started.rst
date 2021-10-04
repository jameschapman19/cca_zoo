Getting Started
===============

cca-zoo is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data.
Where possible I have followed the scikit-learn/mvlearn APIs and models therefore have
fit/transform/fit_transform methods as standard.

Look how easy it is to use:

.. sourcecode:: python

   from cca_zoo.models import CCA
   from cca_zoo.data import generate_covariance_data
   # %%
   (train_view_1,train_view_2),(true_weights_1,true_weights_2)=generate_covariance_data(n=200,view_features=[10,10],latent_dims=1,correlation=1)

   linear_cca = CCA(latent_dims=latent_dims, max_iter=max_iter)

   linear_cca.fit(train_view_1, train_view_2)

In addition to the code snippets in the user guide section, we provide a tutorial notebook hosted on google
colab: https://colab.research.google.com/github/jameschapman19/cca_zoo/blob/master/tutorial_notebooks/cca_zoo_tutorial.ipynb

