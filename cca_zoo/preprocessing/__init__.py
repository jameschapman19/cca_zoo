"""
Class which allows for the different (or the same) processing of multiple views of data.
"""
from mvlearn.utils import check_Xs
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MultiViewPreprocessing(TransformerMixin):
    def __init__(self, preprocessing_list):
        self.preprocessing_list = preprocessing_list

    def fit(self, views, y=None):
        """
        Fits the associated preprocessing steps to each view.
        Parameters
        ----------
        views
        y

        Returns
        -------

        """
        if len(self.preprocessing_list) == 1:
            self.preprocessing_list = self.preprocessing_list * len(views)
        elif len(self.preprocessing_list) != len(views):
            raise ValueError(
                "Length of preprocessing_list must be 1 (apply the same preprocessing to each view) or equal to the number of views"
            )
        check_Xs(views, enforce_views=range(len(self.preprocessing_list)))
        for view, preprocessing in zip(views, self.preprocessing_list):
            # Skip if preprocessing is None
            if preprocessing is not None:
                preprocessing.fit(view, y)
        return self

    def transform(self, X, y=None):
        """
        Transforms each view using the associated preprocessing steps.
        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        [
            check_is_fitted(preprocessing)
            for preprocessing in self.preprocessing_list
            if preprocessing is not None
        ]
        check_Xs(X, enforce_views=range(len(self.preprocessing_list)))
        return [
            # Skip if preprocessing is None
            view if preprocessing is None else preprocessing.transform(view)
            for view, preprocessing in zip(X, self.preprocessing_list)
        ]
