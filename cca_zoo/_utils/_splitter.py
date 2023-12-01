import numpy as np

from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._checks import check_Xs


class SimpleSplitter(TransformerMixin):
    # Authors: Pierre Ablin
    # Copyright (c) 2020 The mvlearn developers.
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    r"""A transformer that splits the features of a single dataset.

    Take a singleview dataset and transform it in a multiview dataset
    by splitting features to different views

    Parameters
    ----------
    n_features : list of ints
        The number of feature to keep in each split: Xs[i] will have shape
        (n_samples, n_features[i])

    Attributes
    ----------
    n_total_features_ : int
        The number of features in the dataset, equal to the sum of n_features_

    n_views_ : int
        The number of views in the output dataset

    See Also
    --------
    ConcatMerger
    """

    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X, y=None):
        r"""Fit to the data.

        Checks that X has a compatible shape.

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y
            Ignored

        Returns
        -------
        self : object
            Transformer instance.
        """
        X = check_array(X)
        _, n_total_features = X.shape
        self.n_total_features_ = sum(self.n_features)
        if self.n_total_features_ != n_total_features:
            raise ValueError(
                "The number of features of X should equal the sum" " of n_features"
            )
        self.n_views_ = len(self.n_features)
        return self

    def transform(self, X, y=None):
        r"""Split data

        The singleview dataset and transform it in a multiview dataset
        by splitting features to different views

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y
            Ignored

        Returns
        -------
        Xs_transformed : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.split(X, np.cumsum(self.n_features)[:-1], axis=1)

    def fit_transform(self, X, y=None):
        r"""Fit to the data and split

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data

        y : array, shape (n_samples,), optional

        Returns
        -------
        Xs_transformed : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Xs):
        r"""Take a multiview dataset and merge it in a single view

        The input dimension must match the fitted dimension of the multiview
        dataset.

        Parameters
        ----------
        Xs : list of numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The input multiview dataset

        Returns
        -------
        X : numpy.ndarray, shape (n_total_features, n_samples)
            The output singleview dataset
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        for X, n_feature in zip(Xs, self.n_features):
            if X.shape[1] != n_feature:
                raise ValueError(
                    "The number of features in Xs does not match" " n_features"
                )

        return np.hstack(Xs)
