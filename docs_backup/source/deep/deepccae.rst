.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Canonically Correlated Autoencoders
===========

.. sourcecode:: python

   from cca_zoo import dccae

   encoder_1 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784)
   encoder_2 = deep_models.Encoder(latent_dims=latent_dims, feature_size=784)
   decoder_1 = deep_models.Decoder(latent_dims=latent_dims, feature_size=784)
   decoder_2 = deep_models.Decoder(latent_dims=latent_dims, feature_size=784)
   dccae_model = dccae.DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])

   # hidden_layer_sizes are shown explicitly but these are also the defaults
   dccae_model = deepwrapper.DeepWrapper(dccae_model)

   dccae_model.fit(train_view_1, train_view_2, epochs=epochs)

   dccae_results = np.stack(
       (dccae_model.train_correlations[0, 1], dccae_model.predict_corr(test_view_1, test_view_2)[0, 1]))

Intro
-----

.. automodule:: cca_zoo.dccae
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource