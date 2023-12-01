import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cca_zoo._utils._checks import check_seaborn_support
from cca_zoo._utils._cross_correlation import cross_corrcoef


class ScoreScatterDisplay:
    """
    Display the scores of a model.

    Args:
        scores (tuple): Tuple of two arrays representing training scores for two representations.
        test_scores (tuple, optional): Tuple of two arrays representing test scores for two representations. Default is None.
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
        scores,
        test_scores=None,
        labels=None,
        test_labels=None,
        show_corr=True,
        ax_labels=None,
        **kwargs,
    ):
        self.scores = scores
        self.test_scores = test_scores
        self.labels = labels
        self.test_labels = test_labels
        (
            self.combined_scores_x,
            self.combined_scores_y,
            self.mode_labels,
            self.combined_labels,
        ) = self._combine_scores()
        self.kwargs = kwargs
        self.show_corr = show_corr
        if self.show_corr:
            self.train_corrs, self.test_corrs = self._calculate_correlations()
        self.ax_labels = ax_labels

    def _combine_scores(self):
        if self.test_scores is not None:
            combined_x = np.vstack((self.scores[0], self.test_scores[0]))
            combined_y = np.vstack((self.scores[1], self.test_scores[1]))
            mode_labels = np.array(
                ["Train"] * len(self.scores[0]) + ["Test"] * len(self.test_scores[0])
            )
            if self.labels is not None:
                combined_labels = self.labels
                if self.test_labels is not None:
                    combined_labels = np.concatenate((self.labels, self.test_labels))
            else:
                combined_labels = None
        else:
            combined_x = self.scores[0]
            combined_y = self.scores[1]
            mode_labels = None
            combined_labels = self.labels
        return combined_x, combined_y, mode_labels, combined_labels

    def _calculate_correlations(self):
        train_corrs = np.diag(
            cross_corrcoef(self.scores[0], self.scores[1], rowvar=False)
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
        check_seaborn_support("ScoreScatterDisplay")

    @classmethod
    def from_estimator(
        cls,
        model,
        train_views,
        test_views=None,
        labels=None,
        test_labels=None,
        ax_labels=None,
        show_corr=True,
        **kwargs,
    ):
        """
        Create a ScoreDisplay instance from an estimator and data representations.

        Args:
            model: The estimator model.
            train_views (tuple): Tuple of two arrays representing training data representations.
            test_views (tuple, optional): Tuple of two arrays representing test data representations. Default is None.
            **kwargs: Additional keyword arguments passed to the ScoreDisplay constructor.

        Returns:
            ScoreScatterDisplay: An instance of ScoreDisplay.
        """
        train_scores = model.transform(train_views)
        test_scores = model.transform(test_views) if test_views is not None else None
        return cls.from_scores(
            train_scores,
            test_scores,
            labels=labels,
            test_labels=test_labels,
            ax_labels=ax_labels,
            show_corr=show_corr,
            **kwargs,
        )

    @classmethod
    def from_scores(
        cls,
        train_scores,
        test_scores=None,
        labels=None,
        test_labels=None,
        ax_labels=None,
        show_corr=True,
        **kwargs,
    ):
        """
        Create a ScoreDisplay instance from precomputed scores.

        Args:
            train_scores (tuple): Tuple of two arrays representing training scores for two representations.
            test_scores (tuple, optional): Tuple of two arrays representing test scores for two representations. Default is None.
            **kwargs: Additional keyword arguments passed to the ScoreDisplay constructor.

        Returns:
            ScoreScatterDisplay: An instance of ScoreDisplay.
        """

        return cls(
            train_scores,
            test_scores,
            labels=labels,
            test_labels=test_labels,
            ax_labels=ax_labels,
            show_corr=show_corr,
            **kwargs,
        )

    def _create_plot(self, x, y, hue, alpha=None, palette=None):
        fig, ax = plt.subplots()
        return (
            sns.scatterplot(
                x=x,
                y=y,
                hue=hue,
                alpha=alpha,
                palette=palette,
                ax=ax,
                **self.kwargs,
            ),
            fig,
            ax,
        )

    def plot(self, title=""):
        dimensions = self.scores[0].shape[1]
        self.figures_ = []

        for i in range(dimensions):
            g, fig, ax = self._create_plot(
                x=self.combined_scores_x[:, i],
                y=self.combined_scores_y[:, i],
                hue=self.combined_labels
                if self.combined_labels is not None
                else self.mode_labels,
            )
            if self.ax_labels is not None:
                ax.set_xlabel(self.ax_labels[0])
                ax.set_ylabel(self.ax_labels[1])
            # if g is a jointplot, get the underlying figure
            plt.suptitle(f"{title} Latent Dimension {i + 1}")

            if self.show_corr:
                if self.test_scores is None:
                    ax.text(
                        0.05,
                        0.95,
                        f"Corr: {self.train_corrs[i]:.2f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                    )
                else:
                    ax.text(
                        0.05,
                        0.95,
                        f"Test Corr: {self.test_corrs[i]:.2f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                    )
            # if there is a legend, move it to the bottom right
            if ax.get_legend() is not None:
                sns.move_legend(ax, "lower right")
            plt.tight_layout()
            self.figures_.append(fig)


class JointScoreScatterDisplay(ScoreScatterDisplay):
    def _create_plot(self, x, y, hue=None, palette=None):
        g = sns.jointplot(
            x=x,
            y=y,
            hue=hue,
            **self.kwargs,
        )
        return g, g.fig, g.ax_joint


class SeparateScoreScatterDisplay(ScoreScatterDisplay):
    def plot(self, title=""):
        dimensions = self.scores[0].shape[1]
        self.train_figures_ = []
        self.test_figures_ = []

        for i in range(dimensions):
            g, fig, ax = self._create_plot(
                x=self.scores[0][:, i],
                y=self.scores[1][:, i],
                hue=self.labels,
                palette=sns.color_palette()[1] if self.labels is None else None,
            )
            if self.ax_labels is not None:
                ax.set_xlabel(self.ax_labels[0])
                ax.set_ylabel(self.ax_labels[1])

            if self.show_corr:
                # put correlation as text in top left corner
                ax.text(
                    0.05,
                    0.95,
                    f"Train Corr: {self.train_corrs[i]:.2f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                )
            plt.suptitle(f"{title} Train - Latent Dimension {i + 1}")
            self.train_figures_.append(g)

            g, fig, ax = self._create_plot(
                x=self.test_scores[0][:, i],
                y=self.test_scores[1][:, i],
                hue=self.labels,
                palette=sns.color_palette()[1] if self.labels is None else None,
            )

            if self.ax_labels is not None:
                ax.set_xlabel(self.ax_labels[0])
                ax.set_ylabel(self.ax_labels[1])
            if self.show_corr:
                # put correlation as text in top left corner
                ax.text(
                    0.05,
                    0.95,
                    f"Test Corr: {self.test_corrs[i]:.2f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                )
            plt.suptitle(f"{title} Test - Latent Dimension {i + 1}")
            self.test_figures_.append(g)
        plt.tight_layout()
        return self


class SeparateJointScoreDisplay(SeparateScoreScatterDisplay):
    def _create_plot(self, x, y, hue=None, palette=None):
        g = sns.jointplot(
            x=x,
            y=y,
            hue=hue,
            **self.kwargs,
        )
        return g, g.fig, g.ax_joint


class PairScoreScatterDisplay(ScoreScatterDisplay):
    def plot(self):
        # Put the combined scores into a dataframe with dimension as column names
        x_vars = [f"X{i}" for i in range(self.combined_scores_x.shape[1])]
        y_vars = [f"Y{i}" for i in range(self.combined_scores_y.shape[1])]
        df = pd.DataFrame(self.combined_scores_x, columns=x_vars)
        df = df.join(pd.DataFrame(self.combined_scores_y, columns=y_vars))
        df["Mode"] = self.mode_labels
        # Plot the pairplot
        g = sns.pairplot(df, hue="Mode", x_vars=x_vars, y_vars=y_vars, **self.kwargs)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.scatterplot)
        # Put the correlation coefficients as text in the top left corner of each plot
        for i in range(self.combined_scores_x.shape[1]):
            g.axes[i, i].text(
                0.05,
                0.95,
                f"Corr: {self.train_corrs[i]:.2f}",
                transform=g.axes[i, i].transAxes,
                verticalalignment="top",
            )
        self.figure_ = g.fig
        return self
