import types
from collections import defaultdict

from sklearn.base import BaseEstimator


class ParameterizableDict(defaultdict, BaseEstimator):
	def __init__(self, default=None, index=[], **kwargs):
		if not isinstance(default, types.FunctionType):
			default = lambda k: default
		super(ParameterizableDict, self).__init__(default)
		self.set_params(**{ k: default for k in index })
		self.set_params(**kwargs)

	def set_params(self, **kwargs):
		self.update(**kwargs)
		return self

	def get_params(self, deep=True):
		return self
