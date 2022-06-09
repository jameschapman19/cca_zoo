Deep Models
-------------

.. currentmodule:: cca_zoo.deepmodels
.. autosummary::
   :nosignatures:

   {% for cls in cca_zoo.deepmodels %}
     {{ cls }}
   {% endfor %}

.. automodule:: cca_zoo.deepmodels
   :members:


Deep Objectives
-------------------------------------

.. autoclass:: cca_zoo.deepmodels._objectives.CCA


.. autoclass:: cca_zoo.deepmodels.objectives.MCCA


.. autoclass:: cca_zoo.deepmodels.objectives.GCCA


.. autoclass:: cca_zoo.deepmodels.objectives.TCCA


Callbacks
-------------------------------------

.. autoclass:: cca_zoo.deepmodels._callbacks.CorrelationCallback
    :members:

.. autoclass:: cca_zoo.deepmodels._callbacks.GenerativeCallback
    :members:

Model Architectures
----------------------------------------

.. automodule:: cca_zoo.deepmodels.architectures
