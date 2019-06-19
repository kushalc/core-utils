import argparse
import itertools
import logging
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd

ITEM_FORMAT = '%(asctime)s %(levelname)s %(funcName)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
def initialize_logging(level=logging.INFO, filename=None):
    # NOTE: Reset so that our configuration overrides anything anyone else set
    # possibly during class initialization, etc.
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        for handler in logger.handlers:
            logger.removeHandler(handler)

    kwargs = {
        "format": ITEM_FORMAT,
        "datefmt": DATE_FORMAT,
        "level": level,
    }
    if filename:
        kwargs["filename"] = filename

    logging.basicConfig(**kwargs)
    logging.captureWarnings(True)
    if level >= logging.INFO:
        warnings.simplefilter("ignore")

    logging.getLogger("bokeh.io.state").setLevel(level + 1)
    return

__old_excepthook__ = sys.excepthook
def handle_unhandled_exception(type, value, tb):
   if hasattr(sys, 'ps1') or \
        not sys.stderr.isatty() or \
        not sys.stdin.isatty() or \
        not sys.stdout.isatty() or \
        type == SyntaxError:
      # we are in interactive mode or we don't have a tty-like
      # device, so we call the default hook
      __old_excepthook__(type, value, tb)
   else:
      # dump the exception and then start the debugger in
      # post-mortem mode.
      logging.critical("Encountered unhandled exception: %s", type, exc_info=(type, value, tb))

      import pdb
      pdb.pm()

TODAY = pd.Timestamp.now(tz="US/Pacific")
def parse_args(description, arguments=[], logging_kwargs={}):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--debug", dest="level", action="store_const", const=logging.DEBUG,
                        default=logging.INFO, help="enable debugging")
    parser.add_argument("--seed", default=hash(TODAY.date), type=int, help="setup initial seed")
    parser.add_argument("--output", default=os.path.join("output", TODAY.strftime("%Y-%m-%d"), parser.prog),
                        help="where to write to")

    for arg in arguments:
        parser.add_argument(arg.pop("name_or_flags"), **arg)

    args = parser.parse_args()
    args.prog = parser.prog
    args.started_at = TODAY

    logging_kwargs = dict(logging_kwargs)
    logging_kwargs["level"] = args.level
    initialize_logging(**logging_kwargs)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # NOTE: Logs every unhandled exception, even without an explicit try-catch.
    # Stop silent failures!
    if os.environ.get("DJANGO_ENV", "development") != "production":
        sys.excepthook = handle_unhandled_exception

    # NOTE: For some reason, there appears to be significant variability in
    # __score between runs separated by an extended period of time. My best
    # guess is that this is due to seeding of the psuedo-RNG. We can either
    # explicitly (a) set the same seed and/or (b) re-compute baseline score
    # before we get started.
    args.seed %= 2**32-1
    try:
        import torch  # naive torch install via pip fails on Heroku, so only set this if torch is explicitly installed.
        torch.manual_seed(args.seed)
    except:
        pass
    np.random.seed(args.seed)
    random.seed(args.seed)

    # other setup
    pd.options.display.max_colwidth = 125

    logging.info("Beginning %s CLI: %s", args.prog, args)
    return args

def first(iterable, default=None):
    item = np.nan
    try:
        for item in iterable:
            return item
    except Exception:
        if pd.notnull(item):
            return item
        else:
            return np.nan

def last(iterable, default=None):
    item = np.nan
    try:
        for item in iterable:
            pass
    except Exception:
        pass
    if pd.notnull(item):
        return item
    else:
        return np.nan

# https://docs.python.org/3/library/itertools.html#itertools.chain.from_iterable
flatten = itertools.chain.from_iterable

def _normalize_X(X, y=None):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Avoid warnings. Also prevents debugging data from getting to caller, but we
    # can figure that out later.
    X = X.copy()
    if y is not None:
        X.loc[:, "specialty_id"] = y
    return X
