import numpy as np

def notempty(df):
    return df.applymap(safe_len) > 0

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
