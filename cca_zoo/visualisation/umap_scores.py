import seaborn as sns

from cca_zoo._utils._checks import check_umap_support, check_seaborn_support
from cca_zoo.visualisation import ScoreScatterDisplay


class UMAPScoreDisplay(ScoreScatterDisplay):
    def _validate_plot_params(self):
        check_umap_support("UMAPScoreDisplay")
        check_seaborn_support("TSNEScoreDisplay")

    def plot(self, **kwargs):
        self._validate_plot_params()
        import umap
        import matplotlib.pyplot as plt

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(self.scores[0])
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=self.labels,
            ax=ax,
            alpha=0.1 if self.test_scores is not None else 1.0,
            label="Train" if self.test_scores is not None else None,
            **self.kwargs,
        )
        if self.test_scores is not None:
            embedding = reducer.transform(self.test_scores[0])
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=self.test_labels,
                ax=ax,
                label="Test",
                **self.kwargs,
            )
        plt.tight_layout()
        self.figure_ = fig
        return self
