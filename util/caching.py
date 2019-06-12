import hashlib
import inspect
import logging
import os
from collections import OrderedDict
from datetime import datetime
from functools import wraps

import cloudpickle

from util.aws import s3_download, s3_exists, s3_path, s3_upload

LOADED_AT = datetime.now()
def _cache_path(module, method, nargs, kwargs, format=None,
                dt=LOADED_AT, basedir=None):
    if not basedir:
        basedir = os.path.join(*[_f for _f in [os.environ.get("APP_ROOT", "."), "tmp",
                                              dt.strftime("%Y-%m-%d") if dt else None] if _f])
    elif dt:
        logging.warn("Ignored dt=%s with basedir", dt)

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
        hashable.update(repr(kwargs).encode())
    except:
        import pdb; pdb.set_trace()

    basename = hashable.hexdigest()
    if format is not None:
        basename += "." + format
    path = os.path.join(basedir, basename)
    return path

# FIXME: DRY up with Estimator._s3_path.
def _s3_path(module, method, nargs, kwargs, format=None, dt=None):
    return _cache_path(module, method, nargs, kwargs, format=format, dt=None,
                       basedir=s3_path([], prefix="curated", dt=dt))

_MEMORY_CACHE = OrderedDict()
def _cache_core(module, method, nargs, kwargs, format,
                use_s3=None, use_disk=True, use_memory=True,
                postfit=None, **cw_kwargs):
    force = kwargs.pop("force", None)  # NOTE: Don't let force affect path.
    if "use_memory" in kwargs:
        use_memory = kwargs.pop("use_memory", use_memory)
    if "use_disk" in kwargs:
        use_disk = kwargs.pop("use_disk", use_disk)
    if "use_s3" in kwargs:
        use_s3 = kwargs.pop("use_s3", use_s3)
        assert(use_disk is not None and use_disk is not False)
    if "postfit" in kwargs:
        postfit = kwargs.pop("postfit", postfit)
    path = _cache_path(module, method, nargs, kwargs, format, **cw_kwargs)
    s3_path = _s3_path(module, method, nargs, kwargs, format, **cw_kwargs)
    if force is not None:
        kwargs["force"] = force
    if postfit is not None:
        kwargs["postfit"] = postfit

    def __postfit(result):
        if postfit is not None:
            postfit(result)
        elif hasattr(result, "_postfit"):
            result._postfit()
        return result

    logging.debug("Looking for %s in memory: %s", method.__name__, path)
    if not force and use_memory and path in _MEMORY_CACHE:
        # Leaving in DEBUG statement in case we want to turn back on. Generates
        # lots of logspam for ResumeParser so disabling for now.
        # logging.debug("Loading %s from memory: %s", method.__name__, path)
        result = _MEMORY_CACHE[path]

    elif not force and use_disk and os.path.exists(path):
        logging.info("Loading %s from disk: %s", method.__name__, path)
        with open(path, "rb") as handle:
            result = __postfit(cloudpickle.load(handle))
        _MEMORY_CACHE[path] = result

    elif not force and use_s3 and s3_exists(s3_path):
        logging.info("Loading %s from S3: %s", method.__name__, s3_path)
        s3_download(s3_path, path)
        with open(path, "rb") as handle:
            result = __postfit(cloudpickle.load(handle))
        _MEMORY_CACHE[path] = result

    else:
        result = method(*nargs, **kwargs)

        logged = False
        if use_s3:
            if not logged:
                logging.info("Caching %s {force=%5.5s} to disk+S3: %s", method.__name__, force, s3_path)
                logged = True
        if use_disk:
            if not logged:
                logging.info("Caching %s {force=%5.5s} to disk: %s", method.__name__, force, path)
                logged = True
            with open(path, "wb") as handle:
                cloudpickle.dump(result, handle)
        if use_s3:
            s3_upload(path, s3_path)
        if use_memory:
            if not logged:
                logging.debug("Caching %s {force=%5.5s} to memory: %s", method.__name__, force, path)
                logged = True
            _MEMORY_CACHE[path] = result

    return result

def _cache_wrapper(method, offset=2, **cw_kwargs):
    caller = inspect.stack()[offset]
    module = inspect.getmodule(caller[0])

    @wraps(method)
    def _wrapper(*nargs, **kwargs):
        return _cache_core(module, method, nargs, kwargs, "cloudpickle",
                           **cw_kwargs)

    return _wrapper

# NOTE: Will continue to use cached result until program is restarted _and_ day
# changes. Put another way, it'll continue to use cached result even if day changes
# until program is restarted (or vice versa).
def cache_locally_today(method):
    return _cache_wrapper(method, use_s3=False, dt=LOADED_AT)
cache_today = cache_locally_today

def cache_everywhere_today(method):
    return _cache_wrapper(method, use_s3=True, dt=LOADED_AT)

# NOTE: If you use cache_s3, please note that it'll re-use it forever until you
# either manually clear the cache or pass in force=True to your method.
def cache_everywhere_forever(method):
    return _cache_wrapper(method, use_s3=True, dt=None)
cache_forever = cache_everywhere_forever
