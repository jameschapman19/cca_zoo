.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Variational Canonical Correlation Analysis
===========

.. sourcecode:: python

   from cca_zoo import dvcca

   encoder_1 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
   encoder_2 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
   decoder_1 = deep_models.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)
   decoder_2 = deep_models.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)
   dvcca_model = dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2],
                             private=False)

   # hidden_layer_sizes are shown explicitly but these are also the defaults
   dvcca_model = deepwrapper.DeepWrapper(dvcca_model)

   dvcca_model.fit(train_view_1, train_view_2, epochs=epochs)

   dvcca_model_results = np.stack(
       (dvcca_model.train_correlations[0, 1], dvcca_model.predict_corr(test_view_1, test_view_2)[0, 1]))

Intro
-----

.. automodule:: cca_zoo.dvcca
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource