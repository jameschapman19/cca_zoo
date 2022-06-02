Deep Models
===========================

DCCA
-------------------------------

.. autoclass:: cca_zoo.deepmodels._dcca.DCCA
    :inherited-members: LightningModule, Module


DCCA by Non-Linear Orthogonal Iterations
-----------------------------------------

.. autoclass:: cca_zoo.deepmodels._dcca_noi.DCCA_NOI
    :inherited-members: LightningModule, Module


Deep Canonically Correlated Autoencoders
-----------------------------------------

.. autoclass:: cca_zoo.deepmodels._dccae.DCCAE
    :inherited-members: LightningModule, Module

Deep Tensor CCA
--------------------------------

.. autoclass:: cca_zoo.deepmodels._dtcca.DTCCA
    :inherited-members: LightningModule, Module

Deep Variational CCA
--------------------------------

.. autoclass:: cca_zoo.deepmodels._dvcca.DVCCA
    :inherited-members: LightningModule, Module

Deep CCA by Stochastic Decorrelation Loss
-----------------------------------------------

.. autoclass:: cca_zoo.deepmodels._dcca_sdl.DCCA_SDL
    :inherited-members: LightningModule, Module

Deep CCA by Barlow Twins
--------------------------------

.. autoclass:: cca_zoo.deepmodels._dcca_barlow_twins.BarlowTwins
    :inherited-members: LightningModule, Module

Split Autoencoders
----------------------------------

.. autoclass:: cca_zoo.deepmodels._splitae.SplitAE
    :inherited-members: LightningModule, Module

Deep Objectives
-------------------------------------

.. autoclass:: cca_zoo.deepmodels._objectives.CCA


.. autoclass:: cca_zoo.deepmodels._objectives.MCCA


.. autoclass:: cca_zoo.deepmodels._objectives.GCCA


.. autoclass:: cca_zoo.deepmodels._objectives.TCCA


Callbacks
-------------------------------------

.. autoclass:: cca_zoo.deepmodels._callbacks.CorrelationCallback
    :members:

.. autoclass:: cca_zoo.deepmodels._callbacks.GenerativeCallback
    :members:

Model Architectures
----------------------------------------

.. automodule:: cca_zoo.deepmodels._architectures
