Installation
=============

cca-zoo can be installed with python versions >=3.7. At the moment there are two options for installation:

1. Clone `cca-zoo` and create a virtual environment using the requirements
3. Install directly from PyPI release without cloning `cca-zoo`.

Installing with pip
----------------------------------------

Install the current release of ``cca-zoo`` with ``pip``::

    $ pip install cca-zoo

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade cca-zoo

Optional dependencies
----------------------------------------

Since some of the functionality depends on PyTorch and Numpyro which are both large packages and may have difficulties
with windows installation, we do not install them by default. To access these,

* [deep]: ``Deep Learning Based Models``
* [probabilistic]: ``Probabilistic Models``
* [all]: ``Include both Probabilistic and Deep Learning Based Models``

If you wish to use these functions, you must install their required dependencies. These are listed in the package requirements folder with corresponding keyword names for manual installation or can be installed from PyPI by simply calling::

    $ pip install cca-zoo[keyword]

where 'keyword' is from the list above, bracketed.
To upgrade the package and torch requirements::

    $ pip install --upgrade cca-zoo[keyword]

Hardware requirements
---------------------
The ``cca-zoo`` package has no specific hardware requirements but some models may be RAM intensive and deep learning and probabilistic models may benefit from a dedicated GPU

OS Requirements
---------------
This package is supported for *Linux* and *macOS* and can also be run on Windows machines.
