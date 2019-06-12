import logging

import pandas as pd


class OptionalParamsMixin(object):
    # Gracefully degrade to no parameters but use it if it exists.
    @classmethod
    def _params_json(cls, **kwargs):
        try:
            return super(OptionalParamsMixin, cls)._params_json(**kwargs)
        except Exception as exc:
            logging.debug("Defaulting to no-params JSON: %s", cls.name(),
                          exc_info=True)
            return {}

class NoParamsMixin(object):
    @classmethod
    def _params_json(cls, **kwargs):
        return {}

class NoGoldsetMixin(object):
    @classmethod
    def goldset(cls):
        return pd.DataFrame(), None

    @classmethod
    def _goldset_target_key(cls):
        return None
