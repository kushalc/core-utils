import ast
import json
import logging
import os
from importlib import import_module

import numpy as np
import pandas as pd


def vine(data, path):
    try:
        for cmp in path.split("."):
            data = data.get(cmp, {})
            try:
                data = unrepr(data)
            except:
                pass
        if data is {}:
            data = None
        return data
    except Exception:
        return None

def unrepr(x, level=logging.WARN):
    y = np.nan
    try:
        if pd.notnull(x) and x not in ["", "nan"]:
            if not x.startswith("u\"") and not x.startswith("u'"):
                if "\"" not in x:
                    x = "u\"" + x + "\""
                elif "'" not in x:
                    x = "u'" + x + "'"
            y = ast.literal_eval(x)
    except Exception as exc:
        logging.log(level, "Couldn't invert repr: %s: %s: '%s'", exc.__class__.__name__, exc, x)
    return y

def load_json(path_or_str):
    try:
        if os.path.exists(path_or_str):
            return json.load(open(path_or_str))
        else:
            return json.loads(path_or_str)
    except ValueError as exception:
        raise ValueError("Couldn't parse JSON: %s" % path_or_str)

def load_class(fqn):
    module, name = fqn.rsplit(".", 1)
    klass = getattr(import_module(module), name)
    return klass

def fully_qualify_class_name(cls):
    return ".".join([cls.__module__, cls.__name__])
fqcn = fully_qualify_class_name
