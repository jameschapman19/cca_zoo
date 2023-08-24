"""
Test the ability to pickle models
"""

import pickle

from cca_zoo.linear import __all__ as linear_models


def test_pickle():
    for model in linear_models:
        print(f"Testing {model}")
        instance = getattr(__import__("cca_zoo.linear", fromlist=[model]), model)(
            latent_dimensions=1
        )
        instance.weights = [1, 2, 3]
        instance2 = pickle.loads(pickle.dumps(instance))
        assert instance.weights == instance2.weights, f"Failed for model {model}"
