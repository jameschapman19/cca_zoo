from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_fit_params
import numpy as np
from skperm.permutation_tests.permutation_test import PermutationTest


def correlation_scorer(estimator, X_test, y_test):
    zx, zy = estimator.transform(X_test, y_test)
    latent_dims = zx.shape[1]
    corrs = np.diag(np.corrcoef(zx, zy, rowvar=False)[:latent_dims, latent_dims:])
    return corrs


class CCAPermutationTest(PermutationTest):
    """

    """
    def __init__(self, estimator, n_permutations=100, n_jobs=None, random_state=0, verbose=0, fit_params=None,
                 exchangeable_errors=True, is_errors=False, ignore_repeat_rows=False, ignore_repeat_perms=False,
                 wilks=True, lawley_hotelling=True, pillai=True, roy_ii=True, roy_iii=True):
        """

        Parameters
        ----------
        estimator :
            A cca estimator
        n_permutations :
            number of permutations
        n_jobs :
            number of processors to use (permutations are computed in parallel)
        random_state :
            random seed for reproducibility
        verbose :
            whether to print progress
        fit_params :
            parameters used to fit estimator
        exchangeable_errors :
            True/False indicating whether to assume exchangeable errors,
            which allow permutations.
        is_errors :
            True/False indicating whether to assume independent and
            symmetric errors, which allow sign-flippings.
        ignore_repeat_rows :
            True/False indicating whether repeated rows in the design
            should be be ignored. Default is false.
        ignore_repeat_perms :
            True/False indicating whether repeated permutations should
            be ignored. Default is false.
        wilks :
            compute wilks
        lawley_hotelling :
            compute lawley-hotelling
        pillai :
            compute pillai
        roy_ii :
            compute roy
        """
        super().__init__(estimator, n_permutations, n_jobs, random_state, verbose, fit_params, exchangeable_errors,
                         is_errors, ignore_repeat_rows, ignore_repeat_perms, scoring=correlation_scorer)
        self.wilks = wilks
        self.lawley_hotelling = lawley_hotelling
        self.pillai = pillai
        self.roy_ii = roy_ii
        self.roy_iii = roy_iii

    def get_metrics(self, score, permutation_scores):
        self.metrics = {}
        self.metrics['pvalue'] = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (self.n_permutations + 1)
        if self.wilks:
            self.metrics['wilks'] = _wilks(score)
        if self.lawley_hotelling:
            self.metrics['lawley_hotelling'] = _lawley_hotelling(score)
        if self.pillai:
            self.metrics['pillai'] = _pillai(score)
        if self.roy_ii:
            self.metrics['roy_ii'] = _roy_ii(score)


def _wilks(score):
    return np.cumprod(1-score**2)

def _lawley_hotelling(score):
    return np.sum(score**2/(1-score**2))

def _pillai(score):
    return np.sum(score**2)

def _roy_ii(score):
    return np.max(score)/(1+np.max(score))