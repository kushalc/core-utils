import numpy as np
import pandas as pd
import regex as re
from bokeh.models import (DatetimeTickFormatter, FixedTicker, MonthsTicker,
                          Range1d)
from bokeh.plotting import figure
from IPython import display
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def setup_basic_plot(title=None, width=959, height=533, x_axis_type=None,
                     x_axis_label=None, y_axis_label=None):
    fp = figure(title=title, plot_width=width, plot_height=height, x_axis_type=x_axis_type,
                x_axis_label=x_axis_label, y_axis_label=y_axis_label)
    if title is not None:
        fp.title.text_font_size = "16pt"
    return fp

_NAN_RE = re.compile(r"\bnan\b")
def idisplay_df(df, precision=3):
    formatters = {}

    precision_flt = precision
    precision_amt = max(precision - 1, 0)
    precision_pct = max(precision - 2, 0)

    if isinstance(df, pd.Series):
        df = df.to_frame()
    for ocol, dtype in df.dtypes.iteritems():
        if isinstance(ocol, str):
            col = ocol.lower()
            if col.endswith("_pct") or col.endswith("_pr"):
                formatters[ocol] = _pretty_number("{:.%d%%}" % precision_pct, precision_pct + 2)
            elif col.endswith("_rpct"):
                formatters[ocol] = _pretty_number("{:+.%d%%}" % precision_pct, precision_pct + 2)
            elif col.endswith("_ct") or col.startswith("num_"):
                formatters[ocol] = _pretty_number("{:,.0f}")
            elif col.endswith("_id"):
                formatters[ocol] = _pretty_number("{:.0f}")
            elif col.endswith("_dt") or is_datetime(df[ocol]):
                formatters[ocol] = _pretty_date()
            elif col.endswith("_amt") or col.endswith("_price"):
                formatters[ocol] = _pretty_number("${:.%df}" % precision_amt, precision_amt + 1)
            elif col.endswith("_lt"):
                formatters[ocol] = _pretty_list()
            elif col in ["mean", "std"] or col.endswith("%"):  # describe()
                formatters[ocol] = _pretty_number("{:.%df}" % precision_flt, precision_flt)

        if not formatters.get(ocol):
            if dtype == int:
                formatters[ocol] = _pretty_number("{:.0f}")
            elif dtype == float:
                formatters[ocol] = _pretty_number("{:.%df}" % precision_flt, precision_flt)
            elif dtype == object:
                formatters[ocol] = _pretty_object()

    display.display(display.HTML(_NAN_RE.sub("", df.style.format(formatters).render())))

def _pretty_list(formatter=str):
    return lambda x: "<br/>".join(map(formatter, x)) if x not in [np.nan, None] else "\n"

def _pretty_object():
    return lambda x: str(x) if pd.notnull(x) else "\n"

def _pretty_date(formatter="%Y-%m-%d"):
    return lambda x: x.strftime(formatter) if pd.notnull(x) else "\n"

def _pretty_number(formatter="{:.0f}", roundable=0):
    return lambda x: formatter.format(x) if pd.notnull(x) and np.round(x, roundable) != 0 else "\n"

_pretty_num = _pretty_number()
_pretty_dec = _pretty_number("{:.1f}", 1)
_pretty_pct = _pretty_number("{:.1%}", 3)
_pretty_usd = _pretty_number("${:.2f}", 2)
