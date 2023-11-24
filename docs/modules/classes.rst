.. _api_ref:

=============
API Reference
=============

This is the class and function reference of CCA-Zoo. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.
For reference on concepts repeated across the API, see :ref:`glossary`.

:mod:`cca_zoo` Package
=======================

.. automodule:: cca_zoo
    :no-members:
    :no-inherited-members:

.. currentmodule:: cca_zoo

:mod:`linear`: Linear Models
============================

.. automodule:: cca_zoo.linear
    :no-members:
    :no-inherited-members:

Linear Models
--------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   linear.CCA
   linear.PLS
   linear.rCCA
   linear.MCCA
   linear.GCCA
   linear.TCCA
   linear.MPLS
   linear.PCACCA
   linear.SPLS
   linear.ElasticCCA
   linear.SCCA_IPLS
   linear.CCA_EY
   linear.PLS_EY
   linear.CCA_GHA
   linear.PLSStochasticPower
   linear.SCCA_Span
   linear.PRCCA
   linear.GRCCA
   linear.PartialCCA


:mod:`nonparametric`: Non-Parametric Methods
============================================

.. automodule:: cca_zoo.nonparametric
    :no-members:
    :no-inherited-members:

Non-Parametric Methods
----------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   nonparametric.KCCA
   nonparametric.KMCCA
   nonparametric.KGCCA
   nonparametric.KTCCA
   nonparametric.NCCA

:mod:`probabilistic`: Probabilistic Methods
===========================================

.. automodule:: cca_zoo.probabilistic
    :no-members:
    :no-inherited-members:

Probabilistic Methods
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   probabilistic.ProbabilisticCCA
   probabilistic.ProbabilisticPLS
   probabilistic.ProbabilisticRCCA

:mod:`deep`: Deep Methods
===========================================

.. automodule:: cca_zoo.deep
    :no-members:
    :no-inherited-members:

Deep Methods
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   deep.DCCA
   deep.DCCA
   deep.DCCA_GHA
   deep.DCCA_SVD
   deep.DGCCA
   deep.DCCAE
   deep.DCCA_NOI
   deep.DCCA_SDL
   deep.DVCCA
   deep.BarlowTwins
   deep.DTCCA
   deep.DCCA_EY
   deep.SplitAE

:mod:`model_selection`: Model Selection
===========================================

.. automodule:: cca_zoo.model_selection
    :no-members:
    :no-inherited-members:

Model Selection
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV
   model_selection.cross_validate
   model_selection.learning_curve
   model_selection.permutation_test_score

:mod:`visualisation`: Visualisation
===========================================

.. automodule:: cca_zoo.visualisation
    :no-members:
    :no-inherited-members:

Visualisation
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   visualisation.ExplainedVarianceDisplay
   visualisation.ScoreScatterDisplay
   visualisation.ScoreScatterDisplay
   visualisation.JointScoreScatterDisplay
   visualisation.SeparateScoreScatterDisplay
   visualisation.SeparateJointScoreDisplay
   visualisation.PairScoreScatterDisplay
   visualisation.ExplainedCovarianceDisplay
   visualisation.WeightHeatmapDisplay
   visualisation.CorrelationHeatmapDisplay
   visualisation.CovarianceHeatmapDisplay
   visualisation.TSNEScoreDisplay
   visualisation.UMAPScoreDisplay
   visualisation.WeightInferenceDisplay

:mod:`datasets`: Datasets
===========================================

.. automodule:: cca_zoo.datasets
    :no-members:
    :no-inherited-members:

Datasets
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   datasets.simulated.JointData
   datasets.simulated.LatentVariableData
   datasets.toy.load_breast_data,
   datasets.toy.load_split_cifar10_data
   datasets.toy.load_mfeat_data
   datasets.toy.load_split_mnist_data

:mod:`preprocessing`: Preprocessing
===========================================

.. automodule:: cca_zoo.preprocessing
    :no-members:
    :no-inherited-members:

Preprocessing
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   preprocessing.MultiViewPreprocessing

:mod:`sequential`: Sequential Methods
===========================================

.. automodule:: cca_zoo.sequential
    :no-members:
    :no-inherited-members:

Sequential Methods
---------------------
.. currentmodule:: cca_zoo

.. autosummary::
   :toctree: generated/

   sequential.SequentialModel