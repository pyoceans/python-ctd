import matplotlib.pyplot as plt
from pandas_flavor import register_dataframe_method, register_series_method


@register_series_method
@register_dataframe_method
def plot_cast(df, **kw):
    """
    Plot a CTD variable.

    """
    ax = kw.pop("ax", None)
    if not ax:
        ax = plt.axes()
    ax.plot(df.values, df.index, **kw)
    ax.set_ylabel(df.index.name)
    ax.set_xlabel(df.name)
    if not getattr(ax, "y_inverted", False):
        ax.invert_yaxis()
        setattr(ax, "y_inverted", True)
    offset = 0.01
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
    return ax
