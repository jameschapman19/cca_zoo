.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Models
===========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Intro
-----

DeepWrapper
-----------

The deepwrapper provides a general training scheme.

.. automodule:: cca_zoo.deepwrapper
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

DCCA_Base
---------

Models passed to DeepWrapper objects need to be of type DCCA_Base. They require both a 'forward' and an 'update_weights'
method.

.. autoclass:: cca_zoo.dcca.DCCA_Base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Models Implemented
------------------

.. include:: deepcca.rst
.. include:: deepccae.rst
.. include:: deepvcca.rst

Architecture Options
--------------------

We provide a base encoder and decoder class which can be inherited in order to build custom architectures which can be
used

.. automodule:: cca_zoo.deep_models
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource