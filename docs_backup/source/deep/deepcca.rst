.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Canonical Correlation Analysis
===========

.. sourcecode:: python

   from cca_zoo import deepwrapper, objectives, dcca, deep_models
   encoder_1 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784)
   encoder_2 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784)
   dcca_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

   # hidden_layer_sizes are shown explicitly but these are also the defaults
   dcca_model = deepwrapper.DeepWrapper(dcca_model)

   dcca_model.fit(train_view_1, train_view_2, epochs=epochs)

   dcca_results = np.stack((dcca_model.train_correlations[0, 1], dcca_model.predict_corr(test_view_1, test_view_2)[0, 1]))

Intro
-----

.. automodule:: cca_zoo.dcca
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource