from cca_zoo.model_selection import permutation_test_score
from sklearn.utils.validation import check_random_state
from cca_zoo.models import CCA

n = 50
rng = check_random_state(0)
X = rng.rand(n, 4)
Y = rng.rand(n, 5)


def test_permutation_test_score():
    p, A, B, U, V = permutation_test_score(X=X, Y=Y, estimator=CCA, latent_dims=2)