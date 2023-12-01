from cca_zoo._utils._checks import check_arviz_support


class WeightInferenceDisplay:
    """
    Class for displaying inference-related plots.

    Attributes
    ----------
    idata : arviz.InferenceData
        The posterior samples.
    true_features: array-like, optional
        The true features for comparison in the plot, defaults to None.
    num_views: int, optional
        The number of representations, defaults to 2.

    """

    def __init__(self, idata, num_views=2, true_features=None):
        """
        Initialize the WeightInferenceDisplay object.

        Parameters
        ----------
        idata : arviz.InferenceData
            The posterior samples.
        num_views : int, optional
            The number of representations, defaults to 2.
        true_features : array-like, optional
            The true features for comparison in the plot, defaults to None.
        """
        self.idata = idata
        self.true_features = true_features
        self.num_views = num_views

    def _validate_plot_params(self):
        """
        Internal method to validate plotting parameters.
        Currently, it checks if arviz is supported.
        """
        check_arviz_support("CorrelationHeatmapDisplay")

    @classmethod
    def from_estimator(cls, pcca_estimator, true_features=None):
        """
        Class method to create an InferenceDisplay instance from an estimator.

        Parameters
        ----------
        pcca_estimator : object
            The estimator object with an 'mcmc' attribute.
        true_features : array-like, optional
            The true features for comparison in the plot, defaults to None.

        Returns
        -------
        WeightInferenceDisplay
            An InferenceDisplay instance.
        """
        return cls.from_mcmc(pcca_estimator.mcmc, true_features)

    @classmethod
    def from_mcmc(cls, mcmc, true_features=None):
        """
        Class method to create an InferenceDisplay instance from mcmc samples.

        Parameters
        ----------
        mcmc : object
            The mcmc samples.
        true_features : array-like, optional
            The true features for comparison in the plot, defaults to None.

        Returns
        -------
        WeightInferenceDisplay
            An InferenceDisplay instance.
        """
        import arviz as az

        idata = az.from_numpyro(mcmc)
        return cls(idata, 2, true_features)

    def plot(self):
        """
        Plot the posterior distributions of parameters and latent variables.
        Adds true values if they are provided.
        """
        import arviz as az
        import matplotlib.pyplot as plt

        for view in range(self.num_views):
            # Plot the posterior distribution of W_0 parameter (for just the first latent variable).
            # Label the weights_ with their weight index. Make all parameters share x axis.
            trace_plot = az.plot_trace(
                self.idata, var_names=[f"W_{view}"], compact=False, divergences=None
            )

            # For each w in W_0, plot the true value from data.true_features[0]
            for i, ax in enumerate(trace_plot[:, 0]):
                if self.true_features is not None:
                    ax.axvline(
                        self.true_features[view].ravel()[i],
                        color="red",
                        linestyle="--",
                        label="True Value",
                    )
                ax.legend()

            plt.suptitle(f"Posterior Distribution of W_{view}")
            plt.tight_layout()

        plt.show()
