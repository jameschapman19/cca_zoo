Installation
=============

cca-zoo is a Python package for canonical correlation analysis (CCA) and its variants. It can be installed with Python versions >=3.7. There are two options for installation:

1. Install directly from PyPI release without cloning `cca-zoo`.
2. Clone `cca-zoo` and create a virtual environment using the requirements.

Installing with pip
----------------------------------------

You can install the current release of ``cca-zoo`` with ``pip``, a Python package manager::

    $ pip install cca-zoo

To upgrade to a newer release, use the ``--upgrade`` flag::

    $ pip install --upgrade cca-zoo

Optional dependencies
----------------------------------------

Some of the functionality of cca-zoo depends on PyTorch and Numpyro, which are both large packages and may have difficulties
with Windows installation. Therefore, we do not install them by default. To access these features, you need to install their required dependencies. These are grouped into three categories:

* [deep]: ``Deep Learning Based Models``
* [probabilistic]: ``Probabilistic Models``
* [all]: ``Include both Probabilistic and Deep Learning Based Models``

You can install these dependencies from PyPI by using the following command::

    $ pip install cca-zoo[keyword]

where 'keyword' is one of the categories above, enclosed in brackets.
To upgrade the package and the optional dependencies, use the following command::

    $ pip install --upgrade cca-zoo[keyword]

Hardware requirements
---------------------
The ``cca-zoo`` package has no specific hardware requirements, but some models may be RAM intensive and deep learning and probabilistic models may benefit from a dedicated GPU.

OS Requirements
---------------
This package is supported for *Linux* and *macOS*, and can also be run on Windows machines.
