# Import the necessary modules
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from cca_zoo._utils._checks import check_seaborn_support


class ExplainedVarianceDisplay:
    """
    Display the explained variance of the latent variables of the representations.

    Parameters
    ----------
    explained_variance_train : np.ndarray
        The explained variance of the train data.
    explained_variance_test : np.ndarray
        The explained variance of the test data.
    ratio : bool
        Whether to plot the ratio of explained variance or not.
    **kwargs : dict
        Keyword arguments to be passed to the seaborn lineplot.

    Attributes
    ----------
    figure_ : matplotlib.pyplot.figure
        The figure of the plot.

    Examples
    --------
    >>> from cca_zoo.visualisation import ExplainedVarianceDisplay
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from cca_zoo.linear import _MCCALoss
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
    >>> # Train an _MCCALoss Model
    >>> # -------------------
    >>> mcca = _MCCALoss(latent_dimensions=2)
    >>> mcca.fit(representations)
    >>>
    >>> # %%
    >>> # Plotting the Explained Variance
    >>> # ---------------------------------
    >>> ExplainedVarianceDisplay.from_estimator(mcca, representations, test_views=test_views).plot()
    >>> plt.show()

    """

    def __init__(
        self,
        explained_variance_train,
        explained_variance_test=None,
        ratio=True,
        view_labels=None,
        **kwargs,
    ):
        self.explained_variance_train = explained_variance_train
        self.explained_variance_test = explained_variance_test
        self.ratio = ratio
        if view_labels is not None:
            assert len(view_labels) == len(
                self.explained_variance_train
            ), "view_labels must be the same length as train_views"
            assert len(view_labels) == len(
                self.explained_variance_test
            ), "view_labels must be the same length as test_views"
            self.view_labels = view_labels
        else:
            self.view_labels = [
                f"View {i}" for i in range(len(self.explained_variance_train))
            ]
        self.kwargs = kwargs

    def _validate_plot_params(self):
        check_seaborn_support("CorrelationHeatmapDisplay")

    @classmethod
    def from_estimator(
        cls, model, train_views, test_views=None, ratio=True, view_labels=None, **kwargs
    ):
        # explained_variance_train will be a numpy array of shape (latent_dimensions,len(train_views))
        if ratio:
            explained_variance_train = model.explained_variance_ratio(train_views)
        else:
            explained_variance_train = model.explained_variance(train_views)
        if test_views is not None:
            if ratio:
                explained_variance_test = model.explained_variance_ratio(test_views)
            else:
                explained_variance_test = model.explained_variance(test_views)
        else:
            explained_variance_test = None
        if ratio:
            return cls.from_explained_variance_ratio(
                explained_variance_train,
                explained_variance_test,
                view_labels=view_labels,
                **kwargs,
            )
        else:
            return cls.from_explained_variance(
                explained_variance_train,
                explained_variance_test,
                view_labels=view_labels,
                **kwargs,
            )

    @classmethod
    def from_explained_variance(
        cls,
        explained_variance_train,
        explained_variance_test=None,
        view_labels=None,
        **kwargs,
    ):
        return cls(
            explained_variance_train,
            explained_variance_test,
            ratio=False,
            view_labels=view_labels,
            **kwargs,
        )

    @classmethod
    def from_explained_variance_ratio(
        cls,
        explained_variance_train,
        explained_variance_test=None,
        view_labels=None,
        **kwargs,
    ):
        return cls(
            explained_variance_train,
            explained_variance_test,
            ratio=True,
            view_labels=view_labels,
            **kwargs,
        )

    def plot(self, ax=None):
        self._validate_plot_params()
        # Use seaborn lineplot with style='Train' and hue='View' to plot the train and test data
        # Reshape the data so that each row has a 'value', 'view index', and 'train' column
        data = pd.DataFrame(self.explained_variance_train, index=self.view_labels).T
        # Give the index a name so that it can be used as a column later
        data.index.name = "Latent dimension"
        # Melt the dataframe so that each row has a 'value', 'view index', and 'train' column
        data = data.reset_index().melt(
            id_vars="Latent dimension", value_vars=self.view_labels
        )
        data.columns = ["Latent dimension", "View", "value"]
        data["Mode"] = "Train"  # Add a column indicating train data
        if self.explained_variance_test is not None:
            data_test = pd.DataFrame(
                self.explained_variance_test, index=self.view_labels
            ).T
            # Give the index a name so that it can be used as a column later
            data_test.index.name = "Latent dimension"
            # Melt the dataframe so that each row has a 'value', 'view index', and 'train' column
            data_test = data_test.reset_index().melt(
                id_vars="Latent dimension", value_vars=self.view_labels
            )
            data_test.columns = ["Latent dimension", "View", "value"]
            data_test["Mode"] = "Test"  # Add a column indicating train data
            data = pd.concat([data, data_test])  # Concatenate the two dataframes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()
        sns.lineplot(
            data=data,
            x="Latent dimension",
            y="value",
            hue="View",
            style="Mode",
            marker="o",
            ax=ax,
        )
        # Set x-ticks to integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Latent dimension")
        if self.ratio:
            ax.set_ylabel("Explained Variance %")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax.set_ylabel("Explained Variance")
        ax.set_title("Explained Variance")
        plt.tight_layout()
        self.figure_ = fig
        return self
