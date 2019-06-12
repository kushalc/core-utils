import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from util.caching import _s3_path, cache_forever
from util.serialization import load_json


class BaseEstimator(sklearn.base.BaseEstimator):
    __variant__ = None

    # NOTE: You need to override with named parameters in order to benefit from
    # get_params() and set_params() for fit-model.
    def __init__(self):
        pass

    @classmethod
    def name(cls, variant=None, full=False):
        components = [_f for _f in [cls.__name__, variant or cls.__variant__] if _f]
        name = "::".join(components)
        if full:
            name = cls.__module__ + "." + name
        return name

    @classmethod
    def get(cls, force=False, use_s3=True, use_disk=True, use_memory=True, **kwargs):
        # FIXME: Make sure this only runs after memory cache miss.
        model = cls.__get(param_kwargs=kwargs, use_s3=use_s3, use_disk=use_disk, 
                          use_memory=use_memory, force=force)
        model._check_inconsistent_params(**kwargs)
        return model

    @classmethod
    @cache_forever
    def __get(cls, param_kwargs={}, use_s3=True, use_disk=True, use_memory=True, force=False):
        return cls.build(**param_kwargs)

    def _check_inconsistent_params(self, **override_kwargs):
        expected = dict(self._params_json())
        expected.update(**override_kwargs)
        expected_df = pd.DataFrame(expected.values(), index=expected.keys(), columns=["expected"])

        actual = self.get_params()
        actual_df = pd.DataFrame(actual.values(), index=actual.keys(), columns=["actual"])

        params_df = expected_df.merge(actual_df, how="outer", validate="1:1", left_index=True, right_index=True)
        params_df.drop(index=[ix for ix in params_df.index if ix.startswith("__")], inplace=True)
        params_df[pd.isnull(params_df)] = np.nan
        inconsistent_df = params_df[(params_df["actual"] != params_df["expected"]) & pd.notnull(params_df["expected"])]

        if inconsistent_df.shape[0]:
            logging.warn("Found inconsistent %s params:\n%s", self.__class__.__name__, inconsistent_df.to_string(na_rep=""))

        return inconsistent_df

    @classmethod
    def build(cls, tuned=True, fit=True, goldset_kwargs={}, **param_kwargs):
        # FIXME FIXME FIXME: THIS IS A BIG HONKING BUG. Although we have param_kwargs
        # above and we get _params_json() right below, we don't pass them into the
        # constructor. THIS MEANS THAT ANY MANIPULATION WE ASSUME IN THE CONSTRUCTOR
        # DOES NOT EVER HAPPEN if it is a function of non-default parameters.
        model = cls()
        if tuned:
            params = cls._params_json()
            params.update(param_kwargs)
            params.pop("__score", None)
            model.set_params(**params)

        if fit:
            goldset = model.goldset(**goldset_kwargs)
            # not returning X, y as expected
            # This should only happen with KeywordOverlap and TitleSimilarity
            # because their goldsets are cached
            # and it's too much of a hassle to regenerate them
            if len(goldset) != 2:
                logging.warn("goldset method for %s not returning X, y as expected, defaulting y to None", cls.__name__)
                X = goldset
                y = None
            else:
                X, y = goldset
            model = model.fit(X, y)

        return model

    def fit(self, *nargs, **kwargs):
        return self._postfit()

    # NOTE: Must be idempotent given fit().
    def _postfit(self):
        return self

    @classmethod
    def goldset(cls):
        df = pd.read_csv(cls._pkg_path("goldsets", "training.tsv"), sep="\t", encoding="utf-8")
        mask = None
        pruned_df = df
        if cls._goldset_target_key() is not None:
            pruned_df = df[pd.notnull(df[cls._goldset_target_key()])]
        if pruned_df.shape[0] < df.shape[0]:
            pruned_pct = 1.000 - pruned_df.shape[0] / float(df.shape[0])
            if pruned_pct >= 0.250:
                level = logging.CRITICAL
            elif pruned_pct >= 0.100:
                level = logging.ERROR
            else:
                level = logging.WARNING
            logging.log(level, "Dropped {:.1%} of {:} goldset".format(pruned_pct, cls.__name__))

        X = pruned_df
        y = None
        if cls._goldset_target_key() is not None:
            X = pruned_df[[c for c in df if c != cls._goldset_target_key()]]
            y = pruned_df[cls._goldset_target_key()]

        if hasattr(cls, "_preprocess_X"):
            X = cls._preprocess_X(X)
        if hasattr(cls, "_preprocess_y"):
            y = cls._preprocess_X(y)
        return X, y

    @classmethod
    def _goldset_kwargs(cls):
        return {}

    @classmethod
    def _goldset_target_key(cls):
        return NotImplementedError()

    @classmethod
    def _raw_scorer(cls, estimator, X, y):
        return estimator.score(X, y)

    @classmethod
    def _cv_scorer(cls, estimator, X, y_gold, random_state=None):
        y_pred = cls._cv_predict(estimator, X, y_gold, random_state=random_state)
        score = cls._metric_score(y_gold, y_pred)
        return score

    @classmethod
    def _cv_predict(cls, estimator, X, y_gold, random_state=None):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        y_pred = cross_val_predict(estimator, X, y_gold, n_jobs=1, cv=estimator._cv_splitter(random_state=random_state))
        return y_pred

    @classmethod
    def _cv_splitter(cls, n_splits=5, random_state=None):
        # FIXME: shuffle=True may be introducing noise between splits. Turning
        # off causes some weird, unexplained (but reproducible) exception.
        #
        # File "/Users/kushalc/Dropbox/TalentWorks/backend-github/postings/specialty_classifier.py", line 79, in predict
        #   y[mask] = model.predict(O[mask])
        # File "/usr/local/lib/python2.7/site-packages/sklearn/utils/metaestimators.py", line 54, in <lambda>
        #   out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        # File "/usr/local/lib/python2.7/site-packages/sklearn/pipeline.py", line 326, in predict
        #   Xt = transform.transform(Xt)
        # File "/usr/local/lib/python2.7/site-packages/sklearn/pipeline.py", line 766, in transform
        #  for name, trans, weight in self._iter())
        # File "/usr/local/lib/python2.7/site-packages/sklearn/preprocessing/data.py", line 1344, in normalize
        #   estimator='the normalize function', dtype=FLOAT_DTYPES)
        # File "/usr/local/lib/python2.7/site-packages/sklearn/utils/validation.py", line 416, in check_array
        #   context))
        # ValueError: Found array with 0 sample(s) (shape=(0, 172)) while a minimum of 1 is required by the normalize function.
        #
        # NOTE: Switching to RepeatedStratifiedKFold, etc. or anything else that repeats the testset will result in errors
        # given blended use of cross_val_predict and _metric_score above.
        return StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=random_state)

    @classmethod
    def _cache_path(cls, method=None, dt=None, format="cloudpickle", **kwargs):
        if method is None:
            method = cls.get
        cached_path = _s3_path(cls.__module__, method, [cls, cls.__variant__], kwargs,
                               dt=dt, format=format)
        return cached_path

    @classmethod
    def _pkg_path(cls, package, filename, cls_name=None):
        if not cls_name:
            cls_name = cls.name(full=True)
        rpath = os.path.join(cls_name, filename)
        apath = os.path.join(package, rpath)
        return apath

    @classmethod
    def _params_json(cls, name="base"):
        # NOTE: This will barf if it can't find the PARAMS_JSON file. After the
        # spaCy debacle of Feb 2017, this is desired. We only want tuned models.
        return load_json(cls._params_path(name))

    @classmethod
    def _params_path(cls, name="base"):
        return cls._pkg_path("parameters", "%s.json" % name)

    @classmethod
    def _tmp_path(cls):
        path = os.path.join(os.environ["APP_ROOT"], "tmp", datetime.now().strftime("%Y-%m-%d"),
                            cls.__module__ + "." + cls.__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
