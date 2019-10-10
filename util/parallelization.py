import gevent
import multiprocessing as mp
from gevent import monkey
from functools import partial

import numpy as np
import pandas as pd

# http://www.gevent.org/intro.html#monkey-patching
monkey.patch_socket()

def parallel_apply(data, func, direct_apply=False, process_ct=2, parallelization_module="multiprocessing", **kwargs):
    return _parallelize(data, partial(_run_on_subset, func, direct_apply=direct_apply, **kwargs), process_ct)

def _parallelize(data, func, process_ct=2, parallelization_module="multiprocessing"):
    if process_ct > 1:
        if parallelization_module == "multiprocessing":
            pool = mp.Pool(process_ct)
            split_data = np.array_split(data, process_ct)  # returns list of np.ndarrays if input is list
            results = pd.concat(pool.map(func, split_data), sort=False)
            pool.close()
            pool.join()

        elif parallelization_module == "gevent":
            split_data = np.array_split(data, process_ct)  # returns list of np.ndarrays if input is list
            jobs = [gevent.spawn(func, subset) for subset in split_data]
            results = pd.concat([job.get() for job in jobs], sort=False)

        else:
            raise ValueError("Unknown parallelization_module: %s" % parallelization_module)

    else:
        results = func(data)

    return results

def _run_on_subset(func, subset_df_or_s, direct_apply=False, **kwargs):
    if direct_apply:
        return func(subset_df_or_s, **kwargs)

    elif isinstance(subset_df_or_s, pd.DataFrame):
        return subset_df_or_s.apply(func, axis=1, **kwargs)

    elif isinstance(subset_df_or_s, (np.ndarray, list, pd.Series)):
        results = [func(item, **kwargs) for item in subset_df_or_s]

        index = range(len(results))
        if isinstance(subset_df_or_s, (pd.Series, pd.DataFrame)):
            index = subset_df_or_s.index

        if isinstance(results[0], (pd.Series, pd.DataFrame)):
            return pd.concat(results, sort=False)
        elif isinstance(results[0], dict):
            return pd.DataFrame(results, index=index)
        else:
            return pd.Series(results, index=index)

    else:
        raise ValueError("Unknown class: %s" % subset_df_or_s.__class__)
