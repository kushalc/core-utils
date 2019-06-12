import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold

from .base import BaseEstimator


class BaseRegressor(BaseEstimator, RegressorMixin):
    @classmethod
    def _cv_splitter(self, n_splits=5):
        return KFold(shuffle=True, n_splits=n_splits)

    def score(self, X_test, y_test, sample_weight=None):
        y_pred = self.predict(X_test)
        if sample_weight is None and "sample_weight" in X_test:
            sample_weight = X_test["sample_weight"]
        return -self._rmse(y_test, y_pred, sample_weight=sample_weight)

    @classmethod
    def _rmse(cls, y_test, y_pred, **kwargs):
        # Filter out NaNs because metric can't handle them
        nan_mask = pd.notna(y_pred)
        return np.sqrt(metrics.mean_squared_error(y_test[nan_mask], y_pred[nan_mask], **kwargs))
