Getting Started with CCA-Zoo
============================

Embarking on your journey with CCA-Zoo is as simple as a breeze. This guide will walk you through the steps to set up and run your first CCA algorithm.

Installation
------------

Before delving into the CCA world, let's get CCA-Zoo installed.

.. code-block:: bash

   pip install cca-zoo

Quick Dive-In
-------------

Once installed, let's run our first CCA analysis using CCA-Zoo:

.. code-block:: python

   import numpy as np
   from cca_zoo.data.simulated import LinearSimulatedData
   from cca_zoo.models import CCA

   # Generate some synthetic data
   data = LinearSimulatedData(...)
   views = data.sample(...)

   # Run CCA
   model = CCA(...)
   model.fit(views)
   results = model.transform(views)

