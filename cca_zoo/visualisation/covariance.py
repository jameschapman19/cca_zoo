import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cca_zoo._utils._checks import check_seaborn_support


class CovarianceHeatmapDisplay:
    """Covariance Heatmap Display

    Heatmap of the covariances between the latent variables of the representations.

    Parameters
    ----------
    train_covariances : np.ndarray
        The train covariances between representations.
    test_covariances : np.ndarray
        The test covariances between representations.

    Attributes
    ----------
    figure_ : matplotlib.pyplot.figure
        The figure of the plot.

    Examples
    --------
    >>> from cca_zoo.visualisation import CovarianceHeatmapDisplay
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from cca_zoo.linear import MCCA
    >>>
    >>> # Generate Sample Data
    >>> # --------------------
    >>> X = np.random.rand(100, 10)
    >>> Y = np.random.rand(100, 10)
    >>>
    >>> # Splitting the data into training and testing sets
    >>> X_train, X_test = X[:50], X[50:]
    >>> Y_train, Y_test = Y[:50], Y[50:]
    >>>
    >>> representations = [X_train, Y_train]
    >>> test_views = [X_test, Y_test]
    >>>
    >>> # Train an MCCA Model
    >>> # -------------------
    >>> mcca = MCCA(latent_dimensions=2)
    >>> mcca.fit(representations)
    >>>
    >>> # %%
    >>> # Plotting the Covariance Heatmap
    >>> # -------------------------------
    >>> CovarianceHeatmapDisplay.from_estimator(mcca, representations, test_views=test_views).plot()
    >>> plt.show()

    """

    def __init__(self, train_covariances, test_covariances):
        self.train_covariances = train_covariances
        self.test_covariances = test_covariances

    def _validate_plot_params(self):
        check_seaborn_support("CorrelationHeatmapDisplay")

    @classmethod
    def from_estimator(cls, model, train_views, test_views=None):
        train_scores = model.transform(train_views)
        if test_views is not None:
            test_scores = model.transform(test_views)
        else:
            test_scores = None
        train_covariances = np.cov(train_scores[0].T, train_scores[1].T)
        if test_scores is not None:
            test_covariances = np.cov(test_scores[0].T, test_scores[1].T)
        else:
            test_covariances = None
        return cls.from_covariances(train_covariances, test_covariances)

    @classmethod
    def from_covariances(cls, train_covariances, test_covariances=None):
        return cls(train_covariances, test_covariances)

    def plot(self):
        self._validate_plot_params()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(
            self.train_covariances,
            annot=True,
            ax=axs[0],
        )
        if self.test_covariances is not None:
            sns.heatmap(
                self.test_covariances,
                annot=True,
                ax=axs[1],
            )
        axs[0].set_title("Train Covariances")
        axs[1].set_title("Test Covariances")
        # plt.tight_layout()
        self.figure_ = fig
        return self
