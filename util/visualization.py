import numpy as np
import pandas as pd
from bokeh.models import (DatetimeTickFormatter, FixedTicker, MonthsTicker,
                          Range1d)
from bokeh.plotting import figure

def setup_basic_plot(title=None, width=959, height=533,
                     x_axis_label=None, y_axis_label=None):
    fp = figure(title=title, plot_width=width, plot_height=height,
                x_axis_label=x_axis_label, y_axis_label=y_axis_label)
    fp.title.text_font_size = "16pt"
    return fp

def _pretty_list():
    return lambda x: "<br/>".join(x)

def _pretty_object():
    return lambda x: str(x) if pd.notnull(x) else "\n"

def _pretty_date(formatter="%Y-%m-%d"):
    return lambda x: x.strftime(formatter) if pd.notnull(x) else "\n"

def _pretty_number(formatter="{:.0f}", roundable=0):
    return lambda x: formatter.format(x) if pd.notnull(x) and np.round(x, roundable) != 0 else "\n"

_pretty_num = _pretty_number()
_pretty_dec = _pretty_number("{:.1f}", 1)
_pretty_pct = _pretty_number("{:.1%}", 1)
_pretty_usd = _pretty_number("${:.2f}", 2)
