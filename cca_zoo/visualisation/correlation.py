import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class CorrelationHeatmapDisplay:
    def __init__(self, train_correlations, test_correlations):
        self.train_correlations = train_correlations
        self.test_correlations = test_correlations

    @classmethod
    def from_estimator(cls, model, train_views, test_views=None):
        train_scores = model.transform(train_views)
        if test_views is not None:
            test_scores = model.transform(test_views)
        else:
            test_scores = None
        train_correlations = np.corrcoef(train_scores[0].T, train_scores[1].T)
        if test_scores is not None:
            test_correlations = np.corrcoef(test_scores[0].T, test_scores[1].T)
        else:
            test_correlations = None
        return cls.from_correlations(train_correlations, test_correlations)

    @classmethod
    def from_correlations(cls, train_correlations, test_correlations=None):
        return cls(train_correlations, test_correlations)

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(
            self.train_correlations,
            annot=True,
            cmap="coolwarm",
            ax=axs[0],
            vmin=-1,
            vmax=1,
        )
        if self.test_correlations is not None:
            sns.heatmap(
                self.test_correlations,
                annot=True,
                cmap="coolwarm",
                ax=axs[1],
                vmin=-1,
                vmax=1,
            )
        axs[0].set_title("Train Correlations")
        axs[1].set_title("Test Correlations")
        plt.tight_layout()
        self.figure_ = fig
        return self
