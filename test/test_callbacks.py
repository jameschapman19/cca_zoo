from sklearn.utils import check_random_state

from cca_zoo.linear import CCAEY, SCCA_PMD, ElasticCCA

n = 50
rng = check_random_state(0)
X = rng.rand(n, 10)
Y = rng.rand(n, 11)
Z = rng.rand(n, 12)
# centre the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)


def test_tracking():
    ey = CCAEY(latent_dimensions=2, track="loss").fit([X, Y])
