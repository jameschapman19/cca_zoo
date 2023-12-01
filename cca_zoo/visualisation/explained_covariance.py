# Import the necessary modules
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from cca_zoo._utils._checks import check_seaborn_support


class ExplainedCovarianceDisplay:
    """
    Display the explained covariance of the latent variables of the representations.

    Parameters
    ----------
    explained_covariance_train : np.ndarray
        The explained covariance of the train data.
    explained_covariance_test : np.ndarray
        The explained covariance of the test data.
    ratio : bool
        Whether to plot the ratio of explained covariance or not.
    **kwargs : dict
        Keyword arguments to be passed to the seaborn lineplot.

    Attributes
    ----------
    figure_ : matplotlib.pyplot.figure
        The figure of the plot.

    Examples
    --------
    >>> from cca_zoo.visualisation import ExplainedCovarianceDisplay
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
    >>> # Plotting the Explained Covariance
    >>> # ---------------------------------
    >>> ExplainedCovarianceDisplay.from_estimator(mcca, representations, test_views=test_views).plot()
    >>> plt.show()

    """

    def __init__(
        self,
        explained_covariance_train,
        explained_covariance_test=None,
        ratio=True,
        **kwargs
    ):
        self.explained_covariance_train = explained_covariance_train
        self.explained_covariance_test = explained_covariance_test
        self.ratio = ratio
        self.kwargs = kwargs

    def _validate_plot_params(self):
        check_seaborn_support("CorrelationHeatmapDisplay")

    @classmethod
    def from_estimator(cls, model, train_views, test_views=None, ratio=True, **kwargs):
        # explained_covariance_train will be a numpy array of shape (latent_dimensions,len(train_views))
        if ratio:
            explained_covariance_train = model.explained_covariance_ratio(train_views)
        else:
            explained_covariance_train = model.explained_covariance(train_views)
        if test_views is not None:
            if ratio:
                explained_covariance_test = model.explained_covariance_ratio(test_views)
            else:
                explained_covariance_test = model.explained_covariance(test_views)
        else:
            explained_covariance_test = None
        if ratio:
            return cls.from_explained_covariance_ratio(
                explained_covariance_train, explained_covariance_test, **kwargs
            )
        else:
            return cls.from_explained_covariance(
                explained_covariance_train, explained_covariance_test, **kwargs
            )

    @classmethod
    def from_explained_covariance(
        cls, explained_covariance_train, explained_covariance_test=None, **kwargs
    ):
        return cls(
            explained_covariance_train, explained_covariance_test, ratio=False, **kwargs
        )

    @classmethod
    def from_explained_covariance_ratio(
        cls, explained_covariance_train, explained_covariance_test=None, **kwargs
    ):
        return cls(
            explained_covariance_train, explained_covariance_test, ratio=True, **kwargs
        )

    def plot(self, ax=None):
        self._validate_plot_params()
        # Use seaborn lineplot with hue='Train' to plot the train and test data
        data = pd.DataFrame(self.explained_covariance_train, columns=["value"])
        data["Mode"] = "Train"  # Add a column indicating train data
        data.index.name = "Latent dimension"
        if self.explained_covariance_test is not None:
            data_test = pd.DataFrame(self.explained_covariance_test, columns=["value"])
            data_test["Mode"] = "Test"  # Add a column indicating test data
            data_test.index.name = "Latent dimension"
            data = pd.concat([data, data_test])  # Concatenate the two dataframes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()
        sns.lineplot(
            data=data, x="Latent dimension", y="value", style="Mode", marker="o", ax=ax
        )
        ax.set_xlabel("Latent dimension")
        if self.ratio:
            ax.set_ylabel("Explained covariance %")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax.set_ylabel("Explained covariance")
        ax.set_title("Explained covariance")
        # Set x-ticks to integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        self.figure_ = fig
        return self
