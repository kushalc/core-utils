import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from sklearn import clone, metrics
from sklearn.model_selection import cross_validate

from util.estimator.tests.base import EstimatorTestMixin


class RegressorTestMixin(EstimatorTestMixin):
    PRECISION = 2

    def test_coverage_and_rmse(self, *nargs):
        logging.info("Loading %s model goldset", self.MODEL_CLS.__name__)
        started_at = pd.Timestamp.now(tz="US/Pacific")
        model, X, y = self._load_model_goldset(*nargs)
        scorers = self._make_scorers(model)

        logging.info("Starting %s validation", self.MODEL_CLS.__name__)
        if hasattr(model, "_goldset_weights"):
            # NOTE: Add `return_train_score = True` to check for overfitting
            scores = cross_val_scores_weighted(model, X, y, model._goldset_weights(),
                                               cv=model._cv_splitter(), scorers=scorers)
        else:
            scores = cross_validate(model, X, y, cv=model._cv_splitter(),
                                    scoring=scorers)

        self._test_best_score(scores["test_coverage"].mean(), method="test_coverage", dt=started_at)

        # NOTE: make_scorer() negates RMSEs to make greater better. Negating back for readability.
        self._test_best_score(-scores["test_rmse"].mean(), method="test_rmse", dt=started_at, greater_is_better=False)
        self._test_best_score(scores["test_equality"].mean(), method="test_equality", dt=started_at)

    @classmethod
    def _make_scorers(cls, model):
        # NOTE: All our metrics ignore NaN values
        # except for coverage which explicitly checks for those
        def _coverage(y_true, y_pred, sample_weight=None, **kwargs):
            if sample_weight is None:
                sample_weight = pd.Series(np.ones(y_true.shape), index=y_true.index)
            numerator = float((pd.notna(y_pred) * sample_weight).sum())
            denominator = sample_weight.sum()
            return numerator / denominator

        def _equality(y_true, y_pred, sample_weight=None, **kwargs):
            nan_mask = pd.notna(y_pred)
            if sample_weight is None:
                sample_weight = pd.Series(np.ones(y_true.shape), index=y_true.index)
            numerator   = ((y_true[nan_mask].round(cls.PRECISION) == y_pred[nan_mask].round(cls.PRECISION)) * sample_weight).sum()
            denominator = sample_weight[nan_mask].sum()
            return numerator / denominator

        scorers = {
            "coverage": metrics.make_scorer(_coverage),
            "rmse": metrics.make_scorer(model._rmse, greater_is_better=False),
            "equality": metrics.make_scorer(_equality)
        }
        return scorers

# Adapted from https://github.com/scikit-learn/scikit-learn/issues/4632#issuecomment-393945555
def cross_val_scores_weighted(model, X, y, weights, cv, scorers):
    scores = defaultdict(list)

    for train_ix, test_ix in cv.split(X, y):
        model_clone = clone(model)
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        weights_train, weights_test = weights.iloc[train_ix], weights.iloc[test_ix]
        model_clone.fit(X_train, y_train, sample_weight=weights_train)

        y_pred = model_clone.predict(X_test)
        for name, scorer in scorers.items():
            score = scorer._sign * scorer._score_func(y_test, y_pred, sample_weight=weights_test)
            scores["test_%s" % name].append(score)

    scores = { name: np.array(results) for name, results in scores.items() }
    return scores
