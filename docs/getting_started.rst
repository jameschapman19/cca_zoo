Getting Started
============================

Embarking on Your Journey
-------------------------

Welcome to the beginning of your exploration with CCA-Zoo. This section serves as a roadmap for initial setup and conducting your first Canonical Correlation Analysis (CCA). Follow these steps to make the most out of CCA-Zoo's comprehensive features.

Installation
------------

Kickstart your experience by installing CCA-Zoo on your machine. Execute the following command to install the package:

.. code-block:: bash

   pip install cca-zoo

After successful installation, you are ready to proceed with your first CCA analysis.

A Step-by-Step Guide to Your First CCA Analysis
-----------------------------------------------

Post-installation, the stage is set for you to engage with CCA. This tutorial will guide you through the basics of applying CCA using synthetic data.

.. code-block:: python

   # Import required libraries
   import numpy as np
   from cca_zoo.data.simulated import LinearSimulatedData
   from cca_zoo.models import CCA

   # Generate synthetic multiview data
   data = LinearSimulatedData(view_features=[10,10],latent_dims: int = 2)
   (X,Y) = data.sample(n_samples=100)

   # Initialize and fit the CCA model
   model = CCA(latent_dimensions=2)
   model.fit(views)

   # Transform the multiview data using the fitted model
   results = model.transform(views)

In this example:

1. We import the necessary modules from NumPy and CCA-Zoo.
2. A synthetic dataset is generated using the `LinearSimulatedData` class.
3. We initialize and fit a CCA model with two components via the `CCA` class.
4. Finally, the fitted model is used to transform the data. The transformed multiview data can be accessed in the `results` variable.

Unveil the Potential: Next Steps
--------------------------------

You've just scratched the surface of what CCA-Zoo can do. Delve deeper into the documentation to explore its extensive functionalities, from advanced model options to multiview learning capabilities. We encourage you to explore, experiment, and contribute to this evolving ecosystem.

