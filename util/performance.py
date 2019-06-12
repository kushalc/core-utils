import time
from functools import wraps

from extern.librato import Librato


def instrument_latency(method):
    @wraps(method)
    def _instrument_latency(*nargs, **kwargs):
        started_at = time.time()
        suffix = ".errors"
        try:
            result = method(*nargs, **kwargs)
            suffix = ".success"
        finally:
            name = "%s::%s" % (nargs[0].__class__.__name__, method.__name__) \
                   if len(nargs) else method.__name__
            latency = time.time() - started_at
            Librato._measure(name, latency)

            # NOTE: Don't alter stack trace if something breaks. If you add an except clause above,
            # it will alter stack trace. Instead, we try to be a little clever about the suffix.
            Librato.count(name + suffix)
        return result

    return _instrument_latency
