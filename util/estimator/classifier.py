
import numpy as np
from sklearn.base import ClassifierMixin

from .base import BaseEstimator


# FIXME: Forking a simpler implementation that avoids complex parameter foo. Merge
# with util.classifier.
class BaseClassifier(BaseEstimator, ClassifierMixin):
    F_BETA = 0.200

    def score(self, X, y_true, sample_weight=None, **kwargs):
        return self._metric_score(X, self.predict(X, **kwargs), sample_weight=sample_weight)

    @classmethod
    def _metric_score(cls, y_true, y_pred, sample_weight=None):
        # FIXME: DRY with build_classification_report.
        from sklearn.metrics import fbeta_score
        return fbeta_score(y_true, y_pred, sample_weight=sample_weight, beta=cls.F_BETA,
                           labels=cls._get_scoreable_labels(), average="weighted")

    def predict(self, X, **kwargs):
        proba_df = self.predict_proba(X, **kwargs)
        y = proba_df.columns[np.argmax(proba_df.values, axis=1)]
        return y.values

    def predict_proba(self, X, **kwargs):
        return np.exp(self.predict_log_proba(X, **kwargs))

    def predict_log_proba(self, X, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _get_scoreable_labels(cls):
        return sorted(set(cls.goldset()[1]))
