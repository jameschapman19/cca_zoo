# Import the necessary modules
"""
Code t
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


class ScoreDisplay:
    """
    Display the scores of a model
    """

    def __init__(self, train_scores, test_scores, **kwargs):
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.kwargs = kwargs

    @classmethod
    def from_estimator(cls, model, train_views,test_views=None, **kwargs):
        train_scores = model.transform(train_views)
        if test_views is not None:
            test_scores = model.transform(test_views)
        else:
            test_scores = None
        return cls.from_scores(train_scores, test_scores, **kwargs)

    @classmethod
    def from_scores(cls, train_scores, test_scores=None, **kwargs):
        return cls(train_scores, test_scores, **kwargs)

    def plot(self):
        dimensions = self.train_scores[0].shape[1]
        # loop through self.train_scores[0].shape[1] and do scatterplots for each dimension of the scores
        for i in range(dimensions):
            fig, ax = plt.subplots(dimensions)
            sns.scatterplot(
                x=self.train_scores[0][:, i],
                y=self.train_scores[1][:, i],
                ax=ax,
                alpha=0.1,
                label="Train",
                **self.kwargs,
            )
            if self.test_scores is not None:
                sns.scatterplot(
                    x=self.test_scores[0][:, i],
                    y=self.test_scores[1][:, i],
                    ax=ax,
                    label="Test",
                    **self.kwargs,
                )
        self.figure_ = fig
        return self


# Define a class that takes the dataset object as an argument
class Plotter:

    def plot_correlation_heatmap(
        self,
        train_scores,
        test_scores=None,
        view_names=None,
        axs=None,
    ):
        # Check the input types and shapes
        assert (
            isinstance(train_scores, list) and len(train_scores) == 2
        ), "train_scores must be a list of two arrays"
        assert (
            train_scores[0].shape[1] == train_scores[1].shape[1]
        ), "train_scores must have the same number of columns"
        if test_scores is not None:
            assert (
                isinstance(test_scores, list) and len(test_scores) == 2
            ), "test_scores must be a list of two arrays"
            assert (
                test_scores[0].shape[1] == test_scores[1].shape[1]
            ), "test_scores must have the same number of columns"
            assert (
                test_scores[0].shape[1] == train_scores[0].shape[1]
            ), "test_scores and train_scores must have the same number of columns"

        # If no axes are given, create new ones
        if axs is None:
            if test_scores is None:
                fig, axs = plt.subplots(figsize=(5, 5))
                axs = [axs]
            else:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Set the default values for optional parameters
        if view_names is None:
            view_names = ["View 1", "View 2"]

        # Compute the correlations for train and test sets
        train_corr = np.corrcoef(train_scores[0].T, train_scores[1].T)[
            : train_scores[0].shape[1], train_scores[0].shape[1] :
        ]
        if test_scores is not None:
            test_corr = np.corrcoef(test_scores[0].T, test_scores[1].T)[
                : train_scores[0].shape[1], train_scores[0].shape[1] :
            ]
            sns.heatmap(
                test_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axs[1]
            )
            axs[1].set_title("Test Correlations")

        # Plot the heatmaps on the given axes
        sns.heatmap(train_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axs[0])

        # Set the titles and labels
        axs[0].set_title("Train Correlations")
        if view_names is not None:
            axs[0].set_xlabel(f"{view_names[0]} scores")
            axs[0].set_ylabel(f"{view_names[1]} scores")
        plt.tight_layout()

        # Return the axes object
        return axs

    def plot_covariance_heatmap(
        self,
        train_scores,
        test_scores,
        axs=None,
    ):
        """Plot the train and test covariances for each dimension as a heatmap.

        Parameters
        ----------
        train_scores : list of two arrays
            The brain and behaviour scores for the train set.
        test_scores : list of two arrays
            The brain and behaviour scores for the test set.
        axs : list of two axes objects, optional
            The axes to plot on. If None, create new axes.

        Returns
        -------
        None
        """
        # If no axes are given, create new ones
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Compute the covariances for train and test sets
        train_cov = np.cov(train_scores[0].T, train_scores[1].T)
        test_cov = np.cov(test_scores[0].T, test_scores[1].T)

        # Plot the heatmaps on the given axes
        sns.heatmap(
            train_cov,
            annot=True,
            cmap="coolwarm",
            ax=axs[0],
        )
        sns.heatmap(
            test_cov,
            annot=True,
            cmap="coolwarm",
            ax=axs[1],
        )

        # Set the titles and labels
        axs[0].set_title("Train Covariances")
        axs[1].set_title("Test Covariances")
        plt.tight_layout()
        return axs

    def plot_weights_heatmap(
        self,
        brain_weights,
        behaviour_weights,
        view_names=None,
        axs=None,
    ):
        """Plot the weights for each dimension as a heatmap.

        Parameters
        ----------
        brain_weights : array
            The brain weights.
        behaviour_weights : array
            The behaviour weights.
        axs : axes object, optional
            The axis to plot on. If None, create a new axis.

        Returns
        -------
        None
        """
        # If no axes are given, create new ones
        # If no axes are given, create new ones
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if view_names is None:
            view_names = ["View 1", "View 2"]
        brain_weights_cov = brain_weights.T @ brain_weights
        behaviour_weights_cov = behaviour_weights.T @ behaviour_weights

        # Plot the heatmap
        sns.heatmap(
            brain_weights_cov,
            annot=True,
            cmap="coolwarm",
            ax=axs[0],
        )
        sns.heatmap(
            behaviour_weights_cov,
            annot=True,
            cmap="coolwarm",
            ax=axs[1],
        )
        axs[0].set_title(f"{view_names[0]} weights")
        axs[1].set_title(f"{view_names[1]} weights")
        plt.tight_layout()
        return axs

    def plot_explained_covariance(
        self,
        model,
        train_views,
        test_views=None,
        ax=None,
        ratio=False,
    ):
        """Plot the explained variance for each dimension.

        Parameters
        ----------
        model : PLSRegression object
            The fitted model to use for computing the explained variance.
        train_views : list of two arrays
            The brain and behaviour data for the train set.
        test_views : list of two arrays
            The brain and behaviour data for the test set.
        ax : axes object, optional
            The axis to plot on. If None, create a new axis.
        ratio : bool, optional
            Whether to plot the explained variance ratio or the absolute value. Default is False.

        Returns
        -------
        None
        """
        # If no axis is given, create a new one
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if ratio:
            explained_cov_train = model.explained_covariance_ratio(train_views)
        else:
            explained_cov_train = model.explained_covariance(train_views)

        # Use seaborn lineplot with hue='Train' to plot the train and test data
        data = pd.DataFrame(explained_cov_train, columns=["value"])
        data["Mode"] = "Train"  # Add a column indicating train data
        data.index.name = "Latent dimension"
        if test_views is not None:
            if ratio:
                explained_cov_test = model.explained_covariance_ratio(test_views)
            else:
                explained_cov_test = model.explained_covariance(test_views)
            data_test = pd.DataFrame(explained_cov_test, columns=["value"])
            data_test["Mode"] = "Test"  # Add a column indicating test data
            data_test.index.name = "Latent dimension"
            data = pd.concat([data, data_test])  # Concatenate the two dataframes
        sns.lineplot(
            data=data, x="Latent dimension", y="value", style="Mode", marker="o", ax=ax
        )

        ax.set_xlabel("Latent dimension")
        if ratio:
            ax.set_ylabel("Explained covariance %")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax.set_ylabel("Explained covariance")
        ax.set_title("Explained covariance")
        plt.tight_layout()
        return ax

    def plot_explained_variance(
        self,
        model,
        train_views,
        test_views=None,
        ax=None,
        view_labels=None,
        ratio=False,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if view_labels is not None:
            assert len(view_labels) == len(
                train_views
            ), "view_labels must be the same length as train_views"
            assert len(view_labels) == len(
                test_views
            ), "view_labels must be the same length as test_views"
        else:
            view_labels = [f"View {i}" for i in range(len(train_views))]

        # explained_variance_train will be a numpy array of shape (latent_dimensions,len(train_views))
        if ratio:
            explained_variance_train = model.explained_variance_ratio(train_views)
        else:
            explained_variance_train = model.explained_variance(train_views)

        # Use seaborn lineplot with style='Train' and hue='View' to plot the train and test data
        # Reshape the data so that each row has a 'value', 'view index', and 'train' column
        data = pd.DataFrame(explained_variance_train, index=view_labels).T
        # Give the index a name so that it can be used as a column later
        data.index.name = "Latent dimension"
        # Melt the dataframe so that each row has a 'value', 'view index', and 'train' column
        data = data.reset_index().melt(
            id_vars="Latent dimension", value_vars=view_labels
        )
        data.columns = ["Latent dimension", "View", "value"]
        data["Mode"] = "Train"  # Add a column indicating train data
        if test_views is not None:
            if ratio:
                explained_variance_test = model.explained_variance_ratio(test_views)
            else:
                explained_variance_test = model.explained_variance(test_views)
            data_test = pd.DataFrame(explained_variance_test, index=view_labels).T
            # Give the index a name so that it can be used as a column later
            data_test.index.name = "Latent dimension"
            # Melt the dataframe so that each row has a 'value', 'view index', and 'train' column
            data_test = data_test.reset_index().melt(
                id_vars="Latent dimension", value_vars=view_labels
            )
            data_test.columns = ["Latent dimension", "View", "value"]
            data_test["Mode"] = "Test"  # Add a column indicating train data
            data = pd.concat([data, data_test])  # Concatenate the two dataframes
        sns.lineplot(
            data=data,
            x="Latent dimension",
            y="value",
            hue="View",
            style="Mode",
            marker="o",
            ax=ax,
        )

        ax.set_xlabel("Latent dimension")
        if ratio:
            ax.set_ylabel("Explained Variance %")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax.set_ylabel("Explained Variance")
        ax.set_title("Explained Variance")
        plt.tight_layout()
        return ax
