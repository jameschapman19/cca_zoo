# Import the necessary modules
"""
Code to generate explained covariance scree plots from cca-zoo models
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


# Define a class that takes the dataset object as an argument
class Plotter:
    def plot_scores_single(
        self,
        train_scores,
        train_labels=None,
        test_scores=None,
        test_labels=None,
        view_names=None,
        title="",
        axs=None,
        **kwargs,
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
        if train_labels is not None:
            assert isinstance(
                train_labels, np.ndarray
            ), "train_labels must be a numpy array"
            assert (
                train_labels.shape[0] == train_scores[0].shape[0]
            ), "train_labels must have the same number of rows as train_scores"
        if test_labels is not None:
            assert isinstance(
                test_labels, np.ndarray
            ), "test_labels must be a numpy array"
            assert (
                test_labels.shape[0] == test_scores[0].shape[0]
            ), "test_labels must have the same number of rows as test_scores"

        # Set the default values for optional parameters
        if axs is None:
            fig, axs = plt.subplots(figsize=(5, 5))
        if train_labels is None:
            train_labels = np.ones(train_scores[0].shape[0])
        if view_names is None:
            view_names = ["View 1", "View 2"]

        # Plot the scatter plot for train and test scores
        hue_order = np.unique(train_labels)
        sns.scatterplot(
            x=train_scores[0],
            y=train_scores[1],
            hue=train_labels,
            ax=axs,
            alpha=0.1,
            label="Train",
            hue_order=hue_order,
            legend=False,
            **kwargs,
        )
        if test_scores is not None:
            if test_labels is None:
                test_labels = np.ones(test_scores[0].shape[0])
            sns.scatterplot(
                x=test_scores[0],
                y=test_scores[1],
                hue=test_labels,
                ax=axs,
                label="Test",
                hue_order=hue_order,
                legend=False,
                **kwargs,
            )

        # axis legend
        handles, labels = axs.get_legend_handles_labels()
        axs.legend(
            handles,
            labels,
        )

        # Set the labels and title

        axs.set_xlabel(f"{view_names[0]} scores")
        axs.set_ylabel(f"{view_names[1]} scores")
        axs.set_title(f"{title}")  # set the title
        plt.tight_layout()

        # Return the axes object
        return axs

    def plot_scores_multi(
        self,
        scores,
        labels=None,
        title="",
        axs=None,
        **kwargs,
    ):
        if labels is None:
            labels = np.ones(scores[0].shape[0])
        data = pd.DataFrame({"Label": labels})
        data["Label"] = data["Label"].astype("category")
        x_vars = [f"view 1 projection {f + 1}" for f in range(scores[0].shape[1])]
        y_vars = [f"view 2 projection {f + 1}" for f in range(scores[1].shape[1])]
        data[x_vars] = scores[0]
        data[y_vars] = scores[1]
        cca_pp = sns.pairplot(data, hue="Label", x_vars=x_vars, y_vars=y_vars)
        cca_pp.fig.suptitle(title)
        # Return the axes object
        return cca_pp

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
