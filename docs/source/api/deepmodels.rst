Deep Models
================================================================================

.. contents:: Contents
    :local:

This module contains classes for various deep learning based CCA models.

Models
--------------------------------------------------------------------------------

This submodule contains classes for different types of deep CCA models, such as discriminative, generative, and self-supervised.

.. automodule:: cca_zoo.deepmodels
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Classes

   .. autosummary::
      :toctree: _autosummary

      DCCA
      DCCAE
      DCCA_NOI
      DCCA_SDL
      DVCCA
      BarlowTwins
      DTCCA
      SplitAE
      DCCA_EY


Objectives
-------------------------------------

This submodule contains classes for different objective functions for deep CCA models.

.. automodule:: cca_zoo.deepmodels.objectives
   :members:
   :show-inheritance:

   .. rubric:: Classes

   .. autosummary::
      :toctree: _autosummary

      CCA
      MCCA
      GCCA
      TCCA


Callbacks
-------------------------------------

This submodule contains classes for different callbacks for deep CCA models including correlation callback.

.. automodule:: cca_zoo.deepmodels.callbacks
   :members:
   :show-inheritance:

   .. rubric:: Classes

   .. autosummary::
      :toctree: _autosummary

      CorrelationCallback


Architectures
----------------------------------------

This submodule contains classes for different architectures for deep CCA models, such as linear, MLP, and CNN.

.. automodule:: cca_zoo.deepmodels.architectures
   :members:
   :show-inheritance:

   .. rubric:: Classes

   .. autosummary::
      :toctree: _autosummary

      Encoder
      Decoder
      CNNEncoder
      CNNDecoder
      LinearEncoder
      LinearDecoder



