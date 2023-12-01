import numpy as np
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate as cross_validate_, check_cv
from sklearn.model_selection import learning_curve as learning_curve_
from sklearn.model_selection._validation import _permutation_test_score, _shuffle
from sklearn.pipeline import Pipeline
from sklearn.utils import indexable, check_random_state
from sklearn.utils.parallel import Parallel, delayed

from cca_zoo._utils._splitter import SimpleSplitter


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

    Parameters
    ----------
    estimator : object
        Estimator object implementing 'fit'. The object to use to fit the data.
    views : list or tuple of array-like
        List or tuple of numpy arrays or array-likes with the same number of rows (samples).
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional, default=None
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), optional, default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a "Group" :term:`cv` instance (e.g., :class:`GroupKFold`).
    scoring : str, callable, list, tuple, or dict, optional, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.
        See notes below for more detail.
    cv : int, cross-validation generator or an iterable, optional, default=None
        Determines the cross-validation splitting strategy. See notes below for more detail.
    n_jobs : int, optional, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, optional, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution. See notes below for more detail.

    Notes
    -----
    For `scoring`:
    If `scoring` represents a single score, one can use:
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.
    If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
    See :ref:`multimetric_grid_search` for an example.

    For `cv`:
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

    For `pre_dispatch`:
    This parameter can be:
        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are
          spawned
        - A str, giving an expression as a function of n_jobs,
          as in '2*n_jobs'
    """

    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([view.shape[1] for view in views])),
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
    Evaluate the significance of a cross-validated score with permutations.

    Permutes targets to generate 'randomized data' and compute the empirical
    p-value against the null hypothesis that features and targets are
    independent. A small p-value suggests that there is a real dependency between
    features and targets which has been used by the estimator to give good predictions.
    A large p-value may be due to lack of real dependency between features and targets
    or the estimator was not able to use the dependency to give good predictions.

    Read more in the :ref:`User Guide <permutation_test_score>`.

    Parameters
    ----------
    estimator : object
        Estimator object implementing 'fit'. The object to use to fit the data.
    views : list or tuple of array-like
        List or tuple of numpy arrays or array-likes with the same number of rows (samples).
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None, optional
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), optional, default=None
        Labels to constrain permutation within groups. When not specified, ``y`` values
        are permuted among all samples. When a grouped cross-validator is used, the
        group labels are also passed on to the ``split`` method of the cross-validator.
    scoring : str or callable, optional, default=None
        A single string (see :ref:`scoring_parameter`) or a callable (see :ref:`scoring`)
        to evaluate the predictions on the test set. If `None` the estimator's score method is used.
    cv : int, cross-validation generator or an iterable, optional, default=None
        Determines the cross-validation splitting strategy. See notes below for more detail.
    n_permutations : int, default=100
        Number of times to permute ``y``.
    n_jobs : int, optional, default=None
        Number of jobs to run in parallel.
    random_state : int, RandomState instance or None, default=0
        Pass an int for reproducible output for permutation of ``y`` values among samples.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, optional, default=None
        Parameters to pass to the fit method of the estimator.

    Notes
    -----
    For `cv`:
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
    """
    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([view.shape[1] for view in views])),
            ("estimator", clone(estimator)),
        ]
    )

    if y is None:
        y = np.zeros(views[0].shape[0])
    X = np.hstack(views)
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator),
            np.hstack((_shuffle(views[0], groups, random_state), *views[1:])),
            y,
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue


def learning_curve(
    estimator,
    X,
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
    Learning Curve.

    Determines cross-validated training and test scores for different training
    set sizes. A cross-validation generator splits the whole dataset k times in
    training and test data. Subsets of the training set with varying sizes will
    be used to train the estimator and a score for each training subset size and
    the test set will be computed. Afterwards, the scores will be averaged over
    all k runs for each training subset size.

    Read more in the :ref:`User Guide <learning_curve>`.

    Parameters
    ----------
    estimator : object
        An object type that implements the "fit" and "predict" methods. An object
        of this type is cloned for each validation.

    representations : list or tuple of numpy arrays or array-likes
        Input data as a list or tuple of numpy arrays or array-likes with the
        same number of rows (samples).

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        Target relative to representations for classification or regression;
        None for unsupervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" cv instance
        (e.g., GroupKFold).

    train_sizes : array-like of shape (n_ticks,), default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e., it has to be within (0, 1].
        Otherwise, it is interpreted as absolute sizes of the training sets.
        Note that for classification, the number of samples usually has to
        be big enough to contain at least one sample from each class.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds in a (Stratified)KFold,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and "y" is
        either binary or multiclass, StratifiedKFold is used. In all other cases,
        KFold is used. These splitters are instantiated with shuffle=False, so
        the splits will be the same across calls. Refer to the
        :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or a scorer callable object /
        function with signature "scorer(estimator, representations, y)".

    exploit_incremental_learning : bool, default=False
        If the estimator supports incremental learning, this will be used to
        speed up fitting for different training set sizes.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the different training and test sets.
        None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors. See the Glossary for more details.

    pre_dispatch : int or str, default='all'
        Number of predispatched jobs for parallel execution (default is all).
        The option can reduce the allocated memory. The str can be an
        expression like '2*n_jobs'.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    shuffle : bool, default=False
        Whether to shuffle training data before taking prefixes of it
        based on "train_sizes".

    random_state : int, RandomState instance, or None, default=None
        Used when "shuffle" is True. Pass an int for reproducible
        output across multiple function calls. See the Glossary for more
        details.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised.

    return_times : bool, default=False
        Whether to return the fit and score times.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    Returns
    -------
    train_sizes_abs : array, shape (n_unique_ticks,)
        Numbers of training examples that have been used to generate the
        learning curve.

    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    fit_times : array, shape (n_ticks, n_cv_folds)
        Times spent for fitting in seconds. Only present if `return_times`
        is True.

    score_times : array, shape (n_ticks, n_cv_folds)
        Times spent for scoring in seconds. Only present if `return_times`
        is True.

    See Also
    --------
    sklearn.model_selection.learning_curve : The function to create the learning
        curve.
    """

    estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([X_.shape[1] for X_ in X])),
            ("estimator", clone(estimator)),
        ]
    )
    return learning_curve_(
        estimator,
        np.hstack(X),
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
