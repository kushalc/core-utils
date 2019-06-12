from bokeh.models import (DatetimeTickFormatter, FixedTicker, MonthsTicker,
                          Range1d)
from bokeh.plotting import figure


def setup_basic_plot(title=None, width=959, height=533,
                     x_axis_label=None, y_axis_label=None):
    fp = figure(title=title, plot_width=width, plot_height=height,
                x_axis_label=x_axis_label, y_axis_label=y_axis_label)
    fp.title.text_font_size = "16pt"
    return fp
