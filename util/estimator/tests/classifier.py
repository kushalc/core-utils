from util.estimator.tests.base import EstimatorTestMixin


class ClassifierTestMixin(EstimatorTestMixin):
    PRECISION = 2

    def test_f_beta(self, *nargs):
        estimator = self.MODEL_CLS.get(force=True, use_s3=False)
        self._test_best_score(estimator._cv_scorer(estimator, *estimator.goldset()))
