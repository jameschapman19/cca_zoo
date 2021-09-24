import numbers
import time
from collections import defaultdict
from itertools import product

import numpy as np
from joblib import Parallel
from sklearn import clone
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._validation import _insert_error_scores
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_random_state, _check_fit_params, _num_samples, indexable

from cca_zoo.model_selection._validation import _fit_and_score


class _MVBaseKFold(_BaseKFold):
    """Base class for multiview KFold, GroupKFold, and StratifiedKFold"""

    def __init__(self, n_splits, *, shuffle, random_state):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, Xs, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        Xs : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        *Xs, y, groups = indexable(*Xs, y, groups)
        n_samples = _num_samples(Xs[0])
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                    .format(self.n_splits, n_samples))

        for train, test in super().split(Xs[0], y, groups):
            yield train, test

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError


class MVKFold(_MVBaseKFold):
    """K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    Read more in the :ref:`User Guide <k_fold>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """

    def __init__(self, n_splits=5, shuffle=False,
                 random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, Xs=None, y=None, groups=None):
        n_samples = _num_samples(Xs[0])
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


def check_cv(cv=5, y=None):
    """Input checker utility for building a cross-validator
    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    y : array-like, default=None
        The target variable for supervised learning problems.
    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        return MVKFold(cv)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        raise ValueError("Expected cv as an integer, cross-validation "
                         "object (from sklearn.model_selection) "
                         "or an iterable. Got %s." % cv)
    return cv


class MVBaseSearchCV(BaseSearchCV):
    def __init__(self, estimator, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs,
                         refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : list of array-likes or numpy.ndarray
            - X length: n_views
            - X[i] shape: (n_samples, n_features_i)
            The data to fit to.
        y : None
            Ignored variable.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
        # TODO
        # X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        # TODO
        cv_orig = check_cv(self.cv, y)
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None,
                                    more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       split_progress=(
                                                           split_idx,
                                                           n_splits),
                                                       candidate_progress=(
                                                           cand_idx,
                                                           n_candidates),
                                                       **fit_and_score_kwargs)
                               for (cand_idx, parameters),
                                   (split_idx, (train, test)) in product(
                    enumerate(candidate_params),
                    enumerate(cv.split(X, y, groups))))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out,
                    all_more_results)

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                        self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _run_search(self, evaluate_candidates):
        """Repeatedly calls `evaluate_candidates` to conduct a search.
        This method, implemented in sub-classes, makes it possible to
        customize the the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.
        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`
            It returns a dict of all results so far, formatted like
            ``cv_results_``.
            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()
        Examples
        --------
        ::
            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")


class MVGridSearchCV(MVBaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.
    Important members are fit, predict.
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.
    Read more in the :ref:`User Guide <grid_search>`.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }
        NOTE
        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        See ``refit`` parameter for more information on allowed values.
    best_score_ : float
        Mean cross-validated score of the best_estimator
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
        This attribute is not available if ``refit`` is a function.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
        .. versionadded:: 0.20
    multimetric_ : bool
        Whether or not the scorers compute several metrics.
    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    """

    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        """
        Parameters
        ----------
        estimator : estimator object.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (`str`) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        scoring : str, callable, list, tuple or dict, default=None
            Strategy to evaluate the performance of the cross-validated model on
            the test set.
            If `scoring` represents a single score, one can use:
            - a single string (see :ref:`scoring_parameter`);
            - a callable (see :ref:`scoring`) that returns a single value.
            If `scoring` represents multiple scores, one can use:
            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric
              names and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.
            See :ref:`multimetric_grid_search` for an example.
        n_jobs : int, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        refit : bool, str, or callable, default=True
            Refit an estimator using the best found parameters on the whole
            dataset.
            For multiple metric evaluation, this needs to be a `str` denoting the
            scorer that would be used to find the best parameters for refitting
            the estimator at the end.
            Where there are considerations other than maximum score in
            choosing a best estimator, ``refit`` can be set to a function which
            returns the selected ``best_index_`` given ``cv_results_``. In that
            case, the ``best_estimator_`` and ``best_params_`` will be set
            according to the returned ``best_index_`` while the ``best_score_``
            attribute will not be available.
            The refitted estimator is made available at the ``best_estimator_``
            attribute and permits using ``predict`` directly on this
            ``GridSearchCV`` instance.
            Also for multiple metric evaluation, the attributes ``best_index_``,
            ``best_score_`` and ``best_params_`` will only be available if
            ``refit`` is set and all of them will be determined w.r.t this specific
            scorer.
            See ``scoring`` parameter to know more about multiple metric
            evaluation.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used. These splitters are instantiated
            with `shuffle=False` so the splits will be the same across calls.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
        verbose : int
            Controls the verbosity: the higher, the more messages.
            - >1 : the computation time for each fold and parameter candidate is
              displayed;
            - >2 : the score is also displayed;
            - >3 : the fold and candidate parameter indexes are also displayed
              together with the starting time of the computation.
        pre_dispatch : int, or str, default=n_jobs
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A str, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'
        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.
        return_train_score : bool, default=False
            If ``False``, the ``cv_results_`` attribute will not include training
            scores.
            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be computationally
            expensive and is not strictly required to select the parameters that
            yield the best generalization performance.
        """
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


def main():
    import itertools
    from cca_zoo.models import rCCA
    x = np.random.rand(1000, 10)
    y = np.random.rand(1000, 10)
    c1 = [0, 0.9, 0.99, 1]
    c2 = [0, 0.9, 0.99, 1]
    param_grid = {'c': list(itertools.product(c1, c2))}
    mod = rCCA()
    mod = MVGridSearchCV(mod, param_grid=param_grid).fit((x, y))
    print()
    print()


if __name__ == '__main__':
    main()
