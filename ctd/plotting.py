import matplotlib.pyplot as plt
import pandas as pd

from pandas_flavor import register_dataframe_method, register_series_method


@register_series_method
@register_dataframe_method
def plot_cast(df, secondary_y=False, label=None, ax=None, *args, **kwargs):
    """
    Plot a CTD variable with the index in the y-axis instead of x-axis.

    """

    fignums = plt.get_fignums()
    if ax is None and not fignums:
        ax = plt.axes()
        fig = ax.get_figure()
    else:
        ax = plt.gca()
        fig = plt.gcf()

    kwargs.setdefault("figsize", (5.25, 6.75))
    figsize = kwargs.pop("figsize")
    fig.set_size_inches(figsize)

    y_inverted = False
    if not getattr(ax, "y_inverted", False):
        setattr(ax, "y_inverted", True)
        y_inverted = True

    if secondary_y:
        ax = ax.twiny()

    xlabel = getattr(df, "name", None)
    ylabel = getattr(df.index, "name", None)

    if isinstance(df, pd.DataFrame):
        labels = label if label else df.columns
        for k, (col, series) in enumerate(df.iteritems()):
            ax.plot(series, series.index, label=labels[k])
    elif isinstance(df, pd.Series):
        label = label if label else str(df.name)
        ax.plot(df.values, df.index, label=label, *args, **kwargs)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if y_inverted and not secondary_y:
        ax.invert_yaxis()
    return ax
