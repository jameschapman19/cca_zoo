from sklearn.utils import check_random_state
from cca_zoo.classical import SCCA_PMD, ElasticCCA
from cca_zoo.classical import CCAEY

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
    pmd = SCCA_PMD(latent_dimensions=2, track="loss", tau=0.5).fit([X, Y])
    elastic = ElasticCCA(latent_dimensions=2, track="loss").fit([X, Y])


def test_convergence():
    elastic = ElasticCCA(
        latent_dimensions=2,
        convergence_checking="loss",
        track="loss",
        alpha=5e-3,
        l1_ratio=0.5,
    ).fit([X, Y])
    pmd = SCCA_PMD(
        latent_dimensions=2,
        convergence_checking="loss",
        track="loss",
        tau=0.5,
    ).fit([X, Y])
