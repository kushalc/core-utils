import inspect
import json
import logging
import tempfile

import numpy as np
import pandas as pd

from extern.librato import Librato
from util.aws import s3_download, s3_upload


class EstimatorTestMixin(object):
    MODEL_CLS = None

    # FIXME: This is a strange default for PRECISION since 0.0
    # means np.power(10.000, -self.PRECISION) == 1.0
    # and we pass if the current score is within 1.0 of the best score
    # which we probably don't want
    # given that most of the time, the score ranges from 0 - 1
    PRECISION = 0
    S3_PREFIX = VALIDATION_PREFIX = "validation"

    @classmethod
    def _current_best_score(cls, method):
        completed_at = np.nan
        score = np.nan
        try:
            # FIXME: Check production environment if last-best file is missing
            # in development-<USER> environment. For now, you'll just have to
            # copy over the file manually.
            result = json.load(open(s3_download(cls._test_s3_path(method)), "r"))
            completed_at = pd.to_datetime(result["completed_at"])
            score = result["score"]
        except:
            logging.warning("Couldn't load current-best %s %s score", cls.MODEL_CLS.name(full=True),
                            method, exc_info=True)

        return score, completed_at

    def _test_best_score(self, current_score, method=None, greater_is_better=True, dt=None):
        if not method:
            method = inspect.stack()[1][3]
        elif hasattr(method, "__name__"):
            method = method.__name__

        if not dt:
            dt = pd.Timestamp.now(tz="US/Pacific")

        result = {
            "completed_at": pd.Timestamp.now(tz="US/Pacific"),
            "score": current_score,
            "started_at": dt,
        }
        formatter = "%%0.%df" % max(self.PRECISION+3, 0)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as handle:
            s3_path = self._test_s3_path(method, dt=dt)
            logging.info("Saving current %%s %%s for tracking: %s: %%s" % formatter,
                         self.MODEL_CLS.name(full=True), method, current_score, s3_path)
            json.dump(result, handle, default=str)
            handle.flush()
            s3_upload(handle.name, s3_path)

        Librato.measure('{}::{}'.format(self.__class__.__name__, method), current_score, unit='')

        best_score, best_at = self._current_best_score(method)
        if pd.notnull(best_score):
            if greater_is_better:
                self.assertGreaterEqual(current_score, best_score - np.power(10.000, -self.PRECISION))
            else:
                self.assertLessEqual(current_score, best_score + np.power(10.000, -self.PRECISION))

        details = { "previous_score": formatter % best_score }
        if pd.notnull(best_at):
            details["previous_at"] = best_at.strftime("%Y-%m-%d %H:%M:%S")
        details["precision"] = self.PRECISION
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as handle:
            logging.info("Saving best(-ish) %%s %%s in history: %s %%s" % formatter,
                         self.MODEL_CLS.name(full=True), method, current_score, details)
            json.dump(result, handle, default=str)
            handle.flush()
            s3_upload(handle.name, self._test_s3_path(method))

    @classmethod
    def _test_s3_path(cls, test_method, dt=None, mode="r"):
        return cls.MODEL_CLS._cache_path(method=test_method, dt=dt, format="json")

    def _load_model_goldset(self, *nargs):
        model = self.MODEL_CLS.get()
        X, y = model.goldset()
        return model, X, y
