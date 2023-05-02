import itertools

import numpy as np
from joblib import Parallel
from mvlearn.compose import SimpleSplitter
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate as cross_validate_, check_cv
from sklearn.model_selection import learning_curve as learning_curve_
from sklearn.pipeline import Pipeline
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_fit_params


def default_scoring(estimator, scoring, views):
    if callable(scoring):
        return scoring
    elif scoring is None:
        if len(views) > 2:

            def scoring(estimator: Pipeline, X, y):
                transformed_views = estimator.transform(X)
                all_corrs = []
                for x, y in itertools.product(transformed_views, repeat=2):
                    all_corrs.append(
                        np.diag(np.corrcoef(x.T, y.T)[: x.shape[1], x.shape[1] :])
                    )
                all_corrs = np.array(all_corrs).reshape(
                    (len(transformed_views), len(transformed_views), x.shape[1])
                )
                return all_corrs

        else:
            scoring = check_scoring(estimator, scoring=scoring)
    return scoring


def cross_validate(
    estimator,
    views,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    """
    Evaluate metric(s) by cross-validation and also record fit/score times.
    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    :param estimator: estimator object implementing 'fit'
        The object to use to fit the data.
    :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
    :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of
        supervised learning.
    :param groups: array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    :param scoring: str, callable, list, tuple, or dict, default=None
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
    :param cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    :param n_jobs: int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param verbose: int, default=0
        The verbosity level.
    :param fit_params: dict, default=None
        Parameters to pass to the fit method of the estimator.
    :param pre_dispatch: int or str, default='2*n_jobs'
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

    :Example:

    """
    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([X_.shape[1] for X_ in views])),
            ("estimator", clone(estimator)),
        ]
    )
    ret = cross_validate_(
        estimator,
        np.hstack(views),
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        error_score=error_score,
    )
    if return_estimator:
        ret["estimator"] = [estimator["estimator"] for estimator in ret["estimator"]]
    return ret


def permutation_test_score(
    estimator,
    views,
    y=None,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    fit_params=None,
):
    """
    Evaluate the significance of a cross-validated score with permutations
    Permutes targets to generate 'randomized data' and compute the empirical
    p-value against the null hypothesis that features and targets are
    independent.
    The p-value represents the fraction of randomized data sets where the
    estimator performed as well or better than in the original data. A small
    p-value suggests that there is a real dependency between features and
    targets which has been used by the estimator to give good predictions.
    A large p-value may be due to lack of real dependency between features
    and targets or the estimator was not able to use the dependency to
    give good predictions.
    Read more in the :ref:`User Guide <permutation_test_score>`.

    :param estimator: estimator object implementing 'fit'
        The object to use to fit the data.
    :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
    :param y: array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    :param groups: array-like of shape (n_samples,), default=None
        Labels to constrain permutation within groups, i.e. ``y`` values
        are permuted among samples with the same group identifier.
        When not specified, ``y`` values are permuted among all samples.
        When a grouped cross-validator is used, the group labels are
        also passed on to the ``split`` method of the cross-validator. The
        cross-validator uses them for grouping the samples  while splitting
        the dataset into train/test set.
    :param scoring: str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If `None` the estimator's score method is used.
    :param cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.
    :param n_permutations: int, default=100
        Number of times to permute ``y``.
    :param n_jobs: int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the cross-validated score are parallelized over the permutations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param random_state: int, RandomState instance or None, default=0
        Pass an int for reproducible output for permutation of
        ``y`` values among samples. See :term:`Glossary <random_state>`.
    :param verbose: int, default=0
        The verbosity level.
    :param fit_params: dict, default=None
        Parameters to pass to the fit method of the estimator.
        .. versionadded:: 0.24

    """
    scorer = default_scoring(estimator, scoring, views)
    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([X_.shape[1] for X_ in views])),
            ("estimator", clone(estimator)),
        ]
    )
    views, y, groups = indexable(np.hstack(views), y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(
        clone(estimator), views, y, groups, cv, scorer, fit_params=fit_params
    )
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator),
            _shuffle(views, groups, random_state, estimator["splitter"]),
            y,
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue


def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params):
    """Auxiliary function for permutation_test_score"""
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        fit_params = _check_fit_params(X, fit_params, train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score, axis=0)


def _shuffle(X, groups, random_state, splitter):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    X = splitter.fit_transform(X)
    for i, X_ in enumerate(X):
        if groups is None:
            indices = random_state.permutation(len(X_))
        else:
            indices = np.arange(len(groups))
            for group in np.unique(groups):
                this_mask = groups == group
                indices[this_mask] = random_state.permutation(indices[this_mask])
        X[i] = _safe_indexing(X_, indices)
    return np.hstack(X)


def learning_curve(
    estimator,
    views,
    y=None,
    groups=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=None,
    scoring=None,
    exploit_incremental_learning=False,
    n_jobs=None,
    pre_dispatch="all",
    verbose=0,
    shuffle=False,
    random_state=None,
    error_score=np.nan,
    return_times=False,
    fit_params=None,
):
    """
    Learning curve.
    Determines cross-validated training and test scores for different training
    set sizes.
    A cross-validation generator splits the whole dataset k times in training
    and test data. Subsets of the training set with varying sizes will be used
    to train the estimator and a score for each training subset size and the
    test set will be computed. Afterwards, the scores will be averaged over
    all k runs for each training subset size.
    Read more in the :ref:`User Guide <learning_curve>`.

    :param estimator: object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
    :param y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to views for classification or regression;
        None for unsupervised learning.
    :param groups: array-like of  shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    :param train_sizes: array-like of shape (n_ticks,), \
            default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
    :param cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    :param scoring: str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, views, y)``.
    :param exploit_incremental_learning: bool, default=False
        If the estimator supports incremental learning, this will be
        used to speed up fitting for different training set sizes.
    :param n_jobs: int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the different training and test sets.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param pre_dispatch: int or str, default='all'
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The str can
        be an expression like '2*n_jobs'.
    :param verbose: int, default=0
        Controls the verbosity: the higher, the more messages.
    :param shuffle: bool, default=False
        Whether to shuffle training data before taking prefixes of it
        based on``train_sizes``.
    :param random_state: int, RandomState instance or None, default=None
        Used when ``shuffle`` is True. Pass an int for reproducible
        output across multiple function calls.
        See :term:`Glossary <random_state>`.
    :param error_score: 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
        .. versionadded:: 0.20
    :param return_times: bool, default=False
        Whether to return the fit and score times.
    :param fit_params: dict, default=None
        Parameters to pass to the fit method of the estimator.
        .. versionadded:: 0.24

    """
    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([X_.shape[1] for X_ in views])),
            ("estimator", clone(estimator)),
        ]
    )
    return learning_curve_(
        estimator,
        np.hstack(views),
        y,
        groups=groups,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        exploit_incremental_learning=exploit_incremental_learning,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        verbose=verbose,
        shuffle=shuffle,
        random_state=random_state,
        error_score=error_score,
        return_times=return_times,
        fit_params=fit_params,
    )
