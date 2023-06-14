from cca_zoo.visualisation import plot_pairwise_scatter, plot_pairwise_correlations
import numpy as np
from cca_zoo.classical import CCA

x = np.random.rand(100, 10)
y = np.random.rand(100, 10)
x -= np.mean(x, axis=0)
y -= np.mean(y, axis=0)


def test_plotting():
    cca = CCA(latent_dimensions=3).fit([x, y])
    plot_pairwise_scatter(cca, [x, y])
    plot_pairwise_correlations(cca, [x, y])
