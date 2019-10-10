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
}
def _handle_disk_cache(path, method, runtime_nargs, runtime_kwargs, format):
    loader, saver = FORMATS[format]
    logging.debug("Looking for %s on disk: %s", method.__name__, path)
    if not runtime_kwargs.get("force", False) and os.path.exists(path):
        logging.info("Loading %s from disk: %s", method.__name__, path)
        with open(path, "rb") as handle:
            result = loader(handle)

    else:
        result = method(*runtime_nargs, **runtime_kwargs)
        logging.info("Caching %s {force=%s} to disk: %s", method.__name__, force, path)
        with open(path, "wb") as handle:
            saver(result, handle)

    return result

def _cache_wrapper(method, offset=2, format="cloudpickle"):
    caller = inspect.stack()[offset]
    module = inspect.getmodule(caller[0])

    @wraps(method)
    def _wrapper(*runtime_nargs, **runtime_kwargs):
        path = _cache_path(module, method, runtime_nargs, runtime_kwargs, format=format)
        result = _handle_disk_cache(path, method, runtime_nargs, runtime_kwargs, format)
        return result

    return _wrapper

# NOTE: Will continue to use cached result until program is restarted _and_ day
# changes. Put another way, it'll continue to use cached result even if day changes
# until program is restarted (or vice versa).
def cache_locally_today(method):
    return _cache_wrapper(method)
cache_today = cache_locally_today
