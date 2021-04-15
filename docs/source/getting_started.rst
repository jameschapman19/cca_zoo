cca-zoo
========

cca-zoo is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data.

Where possible I have followed the scikit-learn/mvlearn APIs and models therefore have fit/transform/fit_transform.

Models can be tuned using a gridsearch.

We provide a tutorial notebook hosted on google colab: https://colab.research.google.com/drive/1reldEBw69hsOtwQOvYsbGGnpvH__b7WF?usp=sharing

Look how easy it is to use:

.. sourcecode:: python

   from cca_zoo import wrappers
   # %%
   linear_cca = wrappers.CCA(latent_dims=latent_dims, max_iter=max_iter)

   linear_cca.fit(train_view_1, train_view_2)

   linear_cca_results = np.stack(
       (linear_cca.train_correlations[0, 1], linear_cca.predict_corr(test_view_1, test_view_2)[0, 1]))