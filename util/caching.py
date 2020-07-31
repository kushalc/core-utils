import glob
import hashlib
import inspect
import logging
import os
from functools import wraps

import cloudpickle
import pandas as pd

LOADED_AT = pd.Timestamp.now()
def _cache_path(module, method, nargs, kwargs, format=None,
                basedir=None, dt=LOADED_AT):
    if not basedir:
        basedir = os.path.join(*filter(None, [".", "tmp", dt.strftime("%Y-%m-%d") if dt else None]))

    def _cls_name(nargs):
        if not nargs:
            return None

        cls = None
        obj = nargs[0]
        if isinstance(obj, type):
            cls = obj
        else:
            cls = obj.__class__

        if cls:
            return cls.__name__

        return None

    basedir = os.path.join(basedir, *[_f for _f in [
        module.__name__ if hasattr(module, "__name__") else str(module),
        _cls_name(nargs),
        method.__name__ if hasattr(method, "__name__") else str(method),
    ] if _f])
    if not basedir.startswith("s3://") and not os.path.exists(basedir):
        os.makedirs(basedir)

    hashable = hashlib.md5()

    try:
        hashable.update(repr(nargs).encode())
        for k in sorted(kwargs.keys()):
            if k in ["force", "use_memory", "use_disk", "use_s3"]:
                continue
            hashable.update(repr(k).encode())
            hashable.update(repr(kwargs[k]).encode())

    except:
        import pdb; pdb.set_trace()

    basename = hashable.hexdigest()
    if format is not None:
        basename += "." + format
    path = os.path.join(basedir, basename)
    return path

FORMATS = {
    "cloudpickle": (cloudpickle.load, cloudpickle.dump),
    "parquet": (pd.read_parquet, pd.DataFrame.to_parquet),
    "feather": (pd.read_feather, pd.DataFrame.to_feather),
}
def _handle_disk_cache(path, method, runtime_nargs, runtime_kwargs, format):
    loader, saver = FORMATS[format]
    globbable_path = path.replace(".parquet", ".*parquet")
    logging.debug("Looking for %s on disk: %s", method.__name__, globbable_path)

    loaded = False
    force = runtime_kwargs.get("force", False)
    if not force:
        try:
            if os.path.exists(path):
                logging.info("Loading %s from disk: %s", method.__name__, path)
                with open(path, "rb") as handle:
                    result = loader(handle)
                    loaded = True

            elif format == "parquet" and glob.glob(globbable_path):
                paths = sorted(glob.glob(globbable_path))
                logging.info("Loading %s from disk: %s", method.__name__, paths)

                result = []
                for path in paths:
                    with open(path, "rb") as handle:
                        result.append(loader(handle))
                loaded = True

        except:
            logging.warn("Couldn't load %s from disk: %s", method.__name__, globbable_path, exc_info=True)
            loaded = False  # just to be safe

    if not loaded:
        result = method(*runtime_nargs, **runtime_kwargs)
        if format == "parquet" and isinstance(result, (tuple, list)):
            assert(len(result) < 1000)
            for ix, rt in enumerate(result):
                with open(path.replace(".parquet", ".{:03d}.parquet".format(ix)), "wb") as handle:
                    logging.info("Caching %s {force=%s} to disk: %s", method.__name__, force, handle.name)
                    saver(rt, handle)
        else:
            with open(path, "wb") as handle:
                logging.info("Caching %s {force=%s} to disk: %s", method.__name__, force, handle.name)
                saver(result, handle)

    return result

# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#decorators-with-arguments
class __CacheWrapper(object):
    def __init__(self, dt=LOADED_AT, cache_format="cloudpickle", stack_offset=1):
        self.cache_format = cache_format
        self.stack_offset = stack_offset
        self.dt = dt

    def __call__(self, method):
        caller = inspect.stack()[self.stack_offset]
        module = inspect.getmodule(caller[0])

        def _wrapper(*runtime_nargs, **runtime_kwargs):
            path = _cache_path(module, method, runtime_nargs, runtime_kwargs, dt=self.dt, format=self.cache_format)
            result = _handle_disk_cache(path, method, runtime_nargs, runtime_kwargs, self.cache_format)
            return result
        return _wrapper

# NOTE: Will continue to use cached result until program is restarted _and_ day
# changes. Put another way, it'll continue to use cached result even if day changes
# until program is restarted (or vice versa).
cache_today = __CacheWrapper()
cache_parquet_today = __CacheWrapper(cache_format="parquet")
cache_feather_today = __CacheWrapper(cache_format="feather")

cache_forever = __CacheWrapper(dt=None)
cache_parquet_forever = __CacheWrapper(dt=None, cache_format="parquet")
cache_feather_forever = __CacheWrapper(dt=None, cache_format="feather")
