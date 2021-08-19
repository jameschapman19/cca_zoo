Getting Started
===============

cca-zoo is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data.

Where possible I have followed the scikit-learn/mvlearn APIs and models therefore have fit/transform/fit_transform.

Models can be tuned using a gridsearch.

We provide a tutorial notebook hosted on google colab: https://colab.research.google.com/github/jameschapman19/cca_zoo/blob/master/tutorial_notebooks/cca_zoo_tutorial.ipynb

Look how easy it is to use:

.. sourcecode:: python

   from cca_zoo.models import CCA
   from cca_zoo.data import generate_covariance_data
   # %%
   (train_view_1,train_view_2),(true_weights_1,true_weights_2)=generate_covariance_data(n=200,view_features=[10,10],latent_dims=1,correlation=1)


   linear_cca = CCA(latent_dims=latent_dims, max_iter=max_iter)

   linear_cca.fit(train_view_1, train_view_2)

   in_sample_correlation =linear_cca.score(train_view_1,train_view_2)
   out_of_sample_correlation =linear_cca.score(test_view_1,test_view_2)

Installation
------------

for everything except the deep learning based models and probabilistic models use: pip install cca-zoo

For deep learning elements use: pip install cca-zoo[deep]
For probabilistic elements use: pip install cca-zoo[probabilistic]

This means that there is no need to install the large pytorch package to run cca-zoo unless you wish to use deep learning
Likewise there is no need to install numpyro to run the standard version of the package