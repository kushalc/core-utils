import logging
import os
import warnings


class Librato(object):
    ACTIVE = len(os.environ.get("LIBRATO_TOKEN", "")) > 0

    @classmethod
    def count(cls, metric, increment = 1):
        if cls.ACTIVE:
            logging.info("count#%s=%d" % (metric, increment))

    @classmethod
    def measure(cls, metric, value, unit = "s"):
        warnings.warn("Prefer @instrument_latency method decorator (vs. Librato.measure)", DeprecationWarning)
        cls._measure(metric, value, unit)

    @classmethod
    def _measure(cls, metric, value, unit = "s"):
        if cls.ACTIVE:
            logging.info("measure#%s=%0.3f%s" % (metric, value, unit))
        elif os.environ.get("DJANGO_ENV", "development") != "production":
            # NOTE: Still print even if Librato is not active for development.
            logging.info("Completed %s: %0.3f%s", metric, value, unit)
