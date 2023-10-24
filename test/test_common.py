import pytest

from sklearn.utils.estimator_checks import check_estimator
from cca_zoo.linear import *

from cca_zoo.linear import __all__ as linear_all

# make a list of all the estimators in the module
estimators = linear_all

# add the estimators to the pytest mark parametrize


@pytest.mark.parametrize(
    "estimator",
    estimators,
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
