import logging

import numpy as np

def not_empty(df):
    return df.applymap(safe_len) > 0

def safety_wrap(method, *nargs, loglevel=logging.NOTSET, **kwargs):
    def _safety_wrapper(value):
        try:
            return method(value, *nargs, **kwargs)
        except:
            logging.log(loglevel, "Couldn't apply %s: %s", method, value)
            return kwargs.get("default", np.nan)

    return _safety_wrapper

def safe_get(dt, key, default=np.nan):
    if dt in [np.nan, None]:
        return default
    value = dt.get(key, default)
    return value

def safe_len(it):
    if it in [np.nan, None]:
        return np.nan
    elif hasattr(it, "__iter__") or hasattr(it, "__getitem__"):
        try:
            return len(it)
        except:
            return np.nan
    else:
        return 1
