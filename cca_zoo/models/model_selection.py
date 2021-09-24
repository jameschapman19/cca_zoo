import operator
from collections.abc import Mapping, Iterable
from functools import partial, reduce
from itertools import product

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_random_state


class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.
    Read more in the :ref:`User Guide <grid_search>`.
    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        An empty dict signifies default parameters.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


class GridSearchCV(BaseSearchCV):
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
            .. versionchanged:: v0.20
               `n_jobs` default changed from 1 to None
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
            .. versionchanged:: 0.20
                Support for callable added.
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
            .. versionchanged:: 0.22
                ``cv`` default value if None changed from 3-fold to 5-fold.
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
            .. versionadded:: 0.19
            .. versionchanged:: 0.21
                Default value was changed from ``True`` to ``False``
        """
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        # evaluate_candidates(ParameterGrid(self.param_grid))
        evaluate_candidates(ParameterGrid(self.param_grid))


class _CrossValidate:
    """
    Base class used for cross validation
    """

    def __init__(self, model, folds: int = 5, verbose: bool = True, random_state=None):
        self.folds = folds
        self.verbose = verbose
        self.model = model
        self.random_state = check_random_state(random_state)

    def score(self, *views: np.ndarray, K=None, **cvparams):
        scores = np.zeros(self.folds)
        inds = np.arange(views[0].shape[0])
        self.random_state.shuffle(inds)
        if self.folds == 1:
            # If 1 fold do an 80:20 split
            fold_inds = np.array_split(inds, 5)
        else:
            fold_inds = np.array_split(inds, self.folds)
        for fold in range(self.folds):
            train_sets = [np.delete(view, fold_inds[fold], axis=0) for view in views]
            val_sets = [view[fold_inds[fold], :] for view in views]
            if K is not None:
                train_obs = np.delete(K, fold_inds[fold], axis=1)
                val_obs = K[:, fold_inds[fold]]
                scores[fold] = self.model.set_params(**cvparams).fit(
                    *train_sets, K=train_obs).score(
                    *val_sets).sum()
            else:
                self.model.set_params(**cvparams).fit(
                    *train_sets)
                scores[fold] = self.model.score(
                    *val_sets).sum()
        scores[np.isnan(scores)] = 0
        std = scores.std(axis=0)
        if self.verbose:
            print(cvparams)
            print(scores.sum(axis=0) / self.folds)
            print(std)
        return scores


def main():
    import itertools
    from cca_zoo.models import PMD
    x = np.random.rand(100, 10)
    y = np.random.rand(100, 8)
    c1 = [1, 3]
    c2 = [1, 3]
    param_grid = {'c': list(itertools.product(c1, c2))}
    pmd = PMD()
    GridSearchCV(pmd, param_grid=param_grid).fit((x, y))
    print()


if __name__ == '__main__':
    main()
