import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class CovarianceHeatmapDisplay:
    def __init__(self, train_covariances, test_covariances):
        self.train_covariances = train_covariances
        self.test_covariances = test_covariances

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
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(
            self.train_covariances,
            annot=True,
            cmap="coolwarm",
            ax=axs[0],
        )
        if self.test_covariances is not None:
            sns.heatmap(
                self.test_covariances,
                annot=True,
                cmap="coolwarm",
                ax=axs[1],
            )
        axs[0].set_title("Train Covariances")
        axs[1].set_title("Test Covariances")
        plt.tight_layout()
        self.figure_ = fig
        return self
