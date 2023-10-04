import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cca_zoo.utils.check_values import check_seaborn_support
import numpy as np

from cca_zoo.utils.cross_correlation import cross_corrcoef


class ScoreDisplay:
    """
    Display the scores of a model.

    Args:
        train_scores (tuple): Tuple of two arrays representing training scores for two views.
        test_scores (tuple, optional): Tuple of two arrays representing test scores for two views. Default is None.
        labels (array-like, optional): Labels for training data. Default is None.
        test_labels (array-like, optional): Labels for test data. Default is None.
        separate (bool, optional): Whether to plot train and test scores separately. Default is False.
        show_corr (bool, optional): Whether to show correlation plots. Default is True.
        **kwargs: Additional keyword arguments passed to seaborn's scatterplot.

    Attributes:
        figure_ (matplotlib.figure.Figure): The generated figure.
    """

    def __init__(
        self,
        train_scores,
        test_scores=None,
        labels=None,
        test_labels=None,
        show_corr=True,
        **kwargs,
    ):
        self.train_scores = train_scores
        self.test_scores = test_scores
        (
            self.combined_scores_x,
            self.combined_scores_y,
            self.mode_labels,
        ) = self._combine_scores()
        self.train_labels = labels
        self.test_labels = test_labels
        # if both train and test labels are not None, combine them
        if self.train_labels is not None and self.test_labels is not None:
            self.combined_labels = np.vstack((self.train_labels, self.test_labels))
        else:
            self.combined_labels = self.mode_labels
        self.kwargs = kwargs
        self.show_corr = show_corr
        if self.show_corr:
            self.train_corrs, self.test_corrs = self._calculate_correlations()

    def _combine_scores(self):
        if self.test_scores is not None:
            combined_x = np.vstack((self.train_scores[0], self.test_scores[0]))
            combined_y = np.vstack((self.train_scores[1], self.test_scores[1]))
            mode_labels = np.array(
                ["Train"] * len(self.train_scores[0])
                + ["Test"] * len(self.test_scores[0])
            )
        else:
            combined_x = self.train_scores[0]
            combined_y = self.train_scores[1]
            mode_labels = None
        return combined_x, combined_y, mode_labels

    def _calculate_correlations(self):
        train_corrs = np.diag(
            cross_corrcoef(self.train_scores[0], self.train_scores[1], rowvar=False)
        )
        test_corrs = None
        if self.test_scores is not None:
            test_corrs = np.diag(
                cross_corrcoef(self.test_scores[0], self.test_scores[1], rowvar=False)
            )
        return train_corrs, test_corrs

    def _validate_plot_params(self):
        """
        Validate plot parameters and check Seaborn support.
        """
        check_seaborn_support("ScoreDisplay")

    @classmethod
    def from_estimator(cls, model, train_views, test_views=None, **kwargs):
        """
        Create a ScoreDisplay instance from an estimator and data views.

        Args:
            model: The estimator model.
            train_views (tuple): Tuple of two arrays representing training data views.
            test_views (tuple, optional): Tuple of two arrays representing test data views. Default is None.
            **kwargs: Additional keyword arguments passed to the ScoreDisplay constructor.

        Returns:
            ScoreDisplay: An instance of ScoreDisplay.
        """
        train_scores = model.transform(train_views)
        test_scores = model.transform(test_views) if test_views is not None else None
        return cls.from_scores(train_scores, test_scores, **kwargs)

    @classmethod
    def from_scores(cls, train_scores, test_scores=None, **kwargs):
        """
        Create a ScoreDisplay instance from precomputed scores.

        Args:
            train_scores (tuple): Tuple of two arrays representing training scores for two views.
            test_scores (tuple, optional): Tuple of two arrays representing test scores for two views. Default is None.
            **kwargs: Additional keyword arguments passed to the ScoreDisplay constructor.

        Returns:
            ScoreDisplay: An instance of ScoreDisplay.
        """

        return cls(train_scores, test_scores, **kwargs)

    def plot(self):
        """
        Plot the scores.

        Returns:
            ScoreDisplay: The ScoreDisplay instance.
        """
        dimensions = self.train_scores[0].shape[1]

        fig, ax = plt.subplots(dimensions)

        for i in range(dimensions):
            sns.scatterplot(
                x=self.combined_scores_x[:, i],
                y=self.combined_scores_y[:, i],
                hue=self.combined_labels,
                ax=ax[i],
                label="Train" if self.test_scores is not None else None,
                **self.kwargs,
            )
            ax[i].set_title(f"Latent Dimension {i+1}")

            if self.show_corr:
                if self.test_scores is None:
                    # put correlation as text in top left corner
                    ax[i].text(
                        0.05,
                        0.95,
                        f"Train Corr: {self.train_corrs[i]:.2f}",
                        transform=ax[i].transAxes,
                        verticalalignment="top",
                    )
                else:
                    # put correlation as text in top left corner
                    ax[i].text(
                        0.05,
                        0.95,
                        f"Test Corr: {self.test_corrs[i]:.2f}",
                        transform=ax[i].transAxes,
                        verticalalignment="top",
                    )
            sns.move_legend(ax[i], "lower right", ncol=2)
        plt.tight_layout()
        self.figure_ = fig
        return self


class SeparateScoreDisplay(ScoreDisplay):
    def plot(self):
        """
        Plot the scores.

        Returns:
            ScoreDisplay: The ScoreDisplay instance.
        """
        dimensions = self.train_scores[0].shape[1]

        fig, ax = plt.subplots(
            dimensions, 2, squeeze=False
        )  # Create two columns for each dimension

        for i in range(dimensions):
            if self.show_corr:
                ax[i, 0].text(
                    0.05,
                    0.95,
                    f"Corr: {self.train_corrs[i]:.2f}",
                    transform=ax[i, 0].transAxes,
                    verticalalignment="top",
                )
            # Plotting training scores
            sns.scatterplot(
                x=self.train_scores[0][:, i],
                y=self.train_scores[1][:, i],
                hue=self.train_labels,
                ax=ax[i, 0],
                **self.kwargs,
            )
            ax[i, 0].set_title(f"Train - Latent Dimension {i+1}")

            # Plotting testing scores if available
            if self.test_scores is not None:
                if self.show_corr:
                    ax[i, 1].text(
                        0.05,
                        0.95,
                        f"Corr: {self.test_corrs[i]:.2f}",
                        transform=ax[i, 1].transAxes,
                        verticalalignment="top",
                    )
                sns.scatterplot(
                    x=self.test_scores[0][:, i],
                    y=self.test_scores[1][:, i],
                    hue=self.test_labels,
                    ax=ax[i, 1],
                    **self.kwargs,
                )
                ax[i, 1].set_title(f"Test - Latent Dimension {i+1}")
        plt.tight_layout()
        self.figure_ = fig
        return self


class JointScoreDisplay(ScoreDisplay):
    def plot(self):
        """
        Plot the scores.

        Returns:
            ScoreDisplay: The ScoreDisplay instance.
        """
        dimensions = self.train_scores[0].shape[1]

        self.figures_ = []

        if self.train_labels is None:
            for i in range(dimensions):
                g = sns.jointplot(
                    x=self.combined_scores_x[:, i],
                    y=self.combined_scores_y[:, i],
                    hue=self.mode_labels,
                    **self.kwargs,
                )
                if self.show_corr:
                    if self.test_scores is None:
                        # put correlation as text in top left corner
                        g.ax_joint.text(
                            0.05,
                            0.95,
                            f"Train Corr: {self.train_corrs[i]:.2f}",
                            transform=g.ax_joint.transAxes,
                            verticalalignment="top",
                        )
                    else:
                        # put correlation as text in top left corner
                        g.ax_joint.text(
                            0.05,
                            0.95,
                            f"Test Corr: {self.test_corrs[i]:.2f}",
                            transform=g.ax_joint.transAxes,
                            verticalalignment="top",
                        )
                g.fig.suptitle(f"Latent Dimension {i+1}")
                self.figures_.append(g.fig)
        return self


class SeparateJointScoreDisplay(SeparateScoreDisplay):
    def plot(self):
        """
        Plot the scores.

        Returns:
            ScoreDisplay: The ScoreDisplay instance.
        """
        dimensions = self.train_scores[0].shape[1]

        self.train_figures_ = []
        self.test_figures_ = []

        for i in range(dimensions):
            # Plotting training scores
            g = sns.jointplot(
                x=self.train_scores[0][:, i],
                y=self.train_scores[1][:, i],
                hue=self.train_labels,
                **self.kwargs,
            )
            if self.show_corr:
                # put correlation as text in top left corner
                g.ax_joint.text(
                    0.05,
                    0.95,
                    f"Corr: {self.train_corrs[i]:.2f}",
                    transform=g.ax_joint.transAxes,
                    verticalalignment="top",
                )
            g.fig.suptitle(f"Train - Latent Dimension {i+1}")
            self.train_figures_.append(g.fig)

            # Plotting testing scores if available
            if self.test_scores is not None:
                g = sns.jointplot(
                    x=self.test_scores[0][:, i],
                    y=self.test_scores[1][:, i],
                    hue=self.test_labels,
                    **self.kwargs,
                )
                if self.show_corr:
                    # put correlation as text in top left corner
                    g.ax_joint.text(
                        0.05,
                        0.95,
                        f"Corr: {self.test_corrs[i]:.2f}",
                        transform=g.ax_joint.transAxes,
                        verticalalignment="top",
                    )
                g.fig.suptitle(f"Test - Latent Dimension {i+1}")
                self.test_figures_.append(g.fig)
        plt.tight_layout()
        return self


class PairScoreDisplay(ScoreDisplay):
    def plot(self):
        # Put the combined scores into a dataframe with dimension as column names
        x_vars = [f"X{i}" for i in range(self.combined_scores_x.shape[1])]
        y_vars = [f"Y{i}" for i in range(self.combined_scores_y.shape[1])]
        df = pd.DataFrame(self.combined_scores_x, columns=x_vars)
        df = df.join(pd.DataFrame(self.combined_scores_y, columns=y_vars))
        df["Mode"] = self.mode_labels
        # Plot the pairplot
        g = sns.pairplot(df, hue="Mode", x_vars=x_vars, y_vars=y_vars, **self.kwargs)
        self.figure_ = g.fig
        return self
