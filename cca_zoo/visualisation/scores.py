import matplotlib.pyplot as plt
import seaborn as sns

from cca_zoo.utils.check_values import check_seaborn_support


class ScoreDisplay:
    """
    Display the scores of a model
    """

    def __init__(
        self,
        train_scores,
        test_scores,
        labels=None,
        test_labels=None,
        separate=False,
        **kwargs,
    ):
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.train_labels = labels
        self.test_labels = test_labels
        self.kwargs = kwargs
        self.separate = separate

    def _validate_plot_params(self):
        check_seaborn_support("CorrelationHeatmapDisplay")

    @classmethod
    def from_estimator(cls, model, train_views, test_views=None, **kwargs):
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

        if self.separate:
            fig, axarr = plt.subplots(
                dimensions, 2
            )  # Create two columns for each dimension

            for i in range(dimensions):
                # Plotting training scores
                sns.scatterplot(
                    x=self.train_scores[0][:, i],
                    y=self.train_scores[1][:, i],
                    hue=self.train_labels,
                    ax=axarr[i, 0],
                    alpha=1.0,
                    label="Train",
                    **self.kwargs,
                )

                axarr[i, 0].set_title(f"Train - Latent Dimension {i+1}")

                # Plotting testing scores if available
                if self.test_scores is not None:
                    sns.scatterplot(
                        x=self.test_scores[0][:, i],
                        y=self.test_scores[1][:, i],
                        hue=self.test_labels,
                        ax=axarr[i, 1],
                        label="Test",
                        **self.kwargs,
                    )

                    axarr[i, 1].set_title(f"Test - Latent Dimension {i+1}")
        else:
            fig, ax = plt.subplots(dimensions)

            for i in range(dimensions):
                sns.scatterplot(
                    x=self.train_scores[0][:, i],
                    y=self.train_scores[1][:, i],
                    hue=self.train_labels,
                    ax=ax[i],
                    alpha=0.1 if self.test_scores is not None else 1.0,
                    label="Train" if self.test_scores is not None else None,
                    **self.kwargs,
                )

                ax[i].set_title(f"Latent Dimension {i+1}")

                if self.test_scores is not None:
                    sns.scatterplot(
                        x=self.test_scores[0][:, i],
                        y=self.test_scores[1][:, i],
                        hue=self.test_labels,
                        ax=ax[i],
                        label="Test",
                        **self.kwargs,
                    )
                    sns.move_legend(ax[i], "lower center", ncol=2)

        plt.tight_layout()
        self.figure_ = fig
        return self
