from sklearn.utils.validation import check_random_state

import cca_zoo.models.innerloop
import cca_zoo.models.innerloop

rng = check_random_state(0)
X = rng.rand(10, 10)
Y = rng.rand(10, 10)
Z = rng.rand(10, 10)


def test_regularized():
    park = cca_zoo.models.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001]).fit(X, Y)
    park_gen = cca_zoo.models.innerloop.ParkhomenkoInnerLoop(c=[0.0001, 0.0001], generalized=True).fit(X,
                                                                                                       Y)
    params = {'c': [2, 2]}
    pmd = cca_zoo.models.innerloop.PMDInnerLoop(c=[2, 2]).fit(X, Y)
    pmd_gen = cca_zoo.models.innerloop.PMDInnerLoop(c=[2, 2], generalized=True).fit(X, Y)
