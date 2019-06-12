from sklearn.base import TransformerMixin

from .base import BaseEstimator
from .mixins import NoGoldsetMixin


# NOTE: NoGoldsetMixin must go *before* BaseEstimator in order to ensure
# their methods are the ones that are called.
class BaseFeaturizer(NoGoldsetMixin, BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # Given a dataframe of user-postings pairs, return an N-dimensional representation
    # of each posting. Most BaseFeaturizers currently return 1-dimensional array of floats,
    # but that's not necessarily required, e.g. FeedbackFeaturizer, DegreeFeaturizer.
    def transform(self, X, y=None):
        raise NotImplementedError()

    def get_feature_names(self):
        raise NotImplementedError()

    # FIXME: DRY.
    @classmethod
    def _pkg_path(cls, package, filename, cls_name=None):
        if not cls_name:
            cls_name = cls.__module__ + "." + cls.__name__
        return BaseEstimator._pkg_path(package, filename, cls_name=cls_name)

    # FIXME: Required for caching.
    def __repr__(self):
        return self.__class__.__name__
