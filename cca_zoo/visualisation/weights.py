import matplotlib.pyplot as plt
import seaborn as sns


class WeightHeatmapDisplay:
    """Heatmap of the weights of a model.

    Parameters
    ----------
    model: CCA model
        A fitted CCA model.
    """

    def __init__(self, weights, view_labels=None, **kwargs):
        self.weights = weights
        self.view_labels = view_labels
        self.kwargs = kwargs

    @classmethod
    def from_estimator(cls, model, view_labels=None, **kwargs):
        weights = model.weights_
        return cls.from_weights(weights, view_labels=view_labels, **kwargs)

    @classmethod
    def from_weights(cls, weights, view_labels=None, **kwargs):
        return cls(weights, view_labels=view_labels, **kwargs)

    def plot(self, **kwargs):
        """Plot the heatmap.

        Parameters
        ----------
        ax: matplotlib axes, optional
            Axes to plot on, by default None.
        kwargs: dict
            Keyword arguments to pass to seaborn.heatmap

        Returns
        -------
        ax: matplotlib axes
            Axes with the heatmap.
        """
        fig, axs = plt.subplots(1, len(self.weights), figsize=(10, 5))
        if self.view_labels is None:
            self.view_labels = [f"View {i}" for i in range(len(self.weights))]
        self.weights_cov = [w.T @ w for w in self.weights]
        # loop through each view and have a heatmap of the covariance of the weights_
        for i, view_weights_cov in enumerate(self.weights_cov):
            sns.heatmap(view_weights_cov, ax=axs[i], annot=True, **self.kwargs)
            axs[i].set_title(self.view_labels[i])
        plt.tight_layout()
        self.figure_ = fig
        return self
