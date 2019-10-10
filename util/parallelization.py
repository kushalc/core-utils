import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd


def parallel_apply(data, func, process_ct=2, direct_apply=False, **kwargs):
    return _parallelize(data, partial(_run_on_subset, func, direct_apply=direct_apply, **kwargs), process_ct)

def _parallelize(df, func, process_ct=2):
    pool = mp.Pool(process_ct)
    split_dfs = np.array_split(df, process_ct)
    data = pd.concat(pool.map(func, split_dfs))
    pool.close()
    pool.join()
    return data

def _run_on_subset(func, subset_df_or_s, direct_apply=False, **kwargs):
    if direct_apply:
        return func(subset_df_or_s, **kwargs)
    if isinstance(subset_df_or_s, pd.DataFrame):
        return subset_df_or_s.apply(func, axis=1, **kwargs)
    elif isinstance(subset_df_or_s, (np.ndarray, pd.Series)):
        results = [func(item, **kwargs) for item in subset_df_or_s]
        if isinstance(results[0], (pd.Series, pd.DataFrame)):
            return pd.concat(results)
        else:
            index = range(len(results))
            if hasattr(subset_df_or_s, "index"):
                index = subset_df_or_s.index
            return pd.Series(results, index=index)
    else:
        raise ValueError("Unknown class: %s" % subset_df_or_s.__class__)
