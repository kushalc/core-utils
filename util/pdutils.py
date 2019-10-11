import numpy as np
import pandas as pd

def notempty(df):
    mask = pd.notnull(df) & \
           df.applymap(safe_len) > 0
    return mask

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
