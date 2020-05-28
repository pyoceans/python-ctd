import numpy as np
import numpy.ma as ma
import pandas as pd

from pandas_flavor import register_dataframe_method, register_series_method


def _rolling_window(data, block):
    """
    http://stackoverflow.com/questions/4936620/
    Using strides for an efficient moving average filter.

    """
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


@register_series_method
@register_dataframe_method
def remove_above_water(df):
    new_df = df.copy()
    return new_df[new_df.index >= 0]


@register_series_method
@register_dataframe_method
def split(df):
    """Returns a tuple with down/up-cast."""
    idx = df.index.argmax() + 1
    down = df.iloc[:idx]
    # Reverse index to orient it as a CTD cast.
    up = df.iloc[idx:][::-1]
    return down, up


@register_series_method
@register_dataframe_method
def lp_filter(df, sample_rate=24.0, time_constant=0.15):
    """
    Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: 911+ systems do not require filter for temperature nor salinity.

    Examples
    --------
    >>> from pathlib import Path
    >>> import matplotlib.pyplot as plt
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> raw = ctd.from_cnv(data_path.joinpath('CTD-spiked-unfiltered.cnv.bz2'))
    >>> prc = ctd.from_cnv(data_path.joinpath('CTD-spiked-filtered.cnv.bz2'))
    >>> kw = {"sample_rate": 24.0, "time_constant": 0.15}
    >>> original = prc.index.values
    >>> unfiltered = raw.index.values
    >>> filtered = raw.lp_filter(**kw).index.values
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(original, 'k', label='original')
    >>> l2, = ax.plot(unfiltered, 'r', label='unfiltered')
    >>> l3, = ax.plot(filtered, 'g', label='filtered')
    >>> leg = ax.legend()

    Notes
    -----
    https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

    """

    from scipy import signal

    # Butter is closer to what SBE is doing with their cosine filter.
    Wn = (1.0 / time_constant) / (sample_rate * 2.0)
    b, a = signal.butter(2, Wn, "low")
    new_df = df.copy()
    new_df.index = signal.filtfilt(b, a, df.index.values)
    return new_df


@register_series_method
@register_dataframe_method
def press_check(df):
    """
    Remove pressure reversals from the index.

    """
    new_df = df.copy()
    press = new_df.copy().index.values

    ref = press[0]
    inversions = np.diff(np.r_[press, press[-1]]) < 0
    mask = np.zeros_like(inversions)
    for k, p in enumerate(inversions):
        if p:
            ref = press[k]
            cut = press[k + 1 :] < ref
            mask[k + 1 :][cut] = True
    new_df[mask] = np.NaN
    return new_df


def _bindata(series, delta, method):
    start = np.ceil(series.index[0])
    stop = np.floor(series.index[-1])
    new_index = np.arange(start, stop, delta)
    binned = pd.cut(series.index, bins=new_index)
    if method == "average":
        new_series = series.groupby(binned).mean()
        new_series.index = new_index[:-1] + delta / 2
    elif method == "interpolate":
        data = np.interp(new_index, series.index, series)
        return pd.Series(data, index=new_index, name=series.name)
    else:
        raise ValueError(
            f"Expected method `average` or `interpolate`, but got {method}."
        )
    return new_series


@register_series_method
@register_dataframe_method
def bindata(df, delta=1.0, method="average"):
    """
    Bin average the index (usually pressure) to a given interval (default
    delta = 1).

    """
    if isinstance(df, pd.Series):
        new_df = _bindata(df, delta=delta, method=method)
    else:
        new_df = df.apply(_bindata, delta=delta, method=method)
    return new_df


def _despike(series, n1, n2, block, keep):
    """
    Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`.

    """

    data = series.values.astype(float).copy()
    roll = _rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = np.abs(data - mean.filled(fill_value=np.NaN)) > std.filled(fill_value=np.NaN)
    data[mask] = np.NaN

    # Pass two recompute the mean and std without the flagged values from pass
    # one and removed the flagged data.
    roll = _rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    values = series.values.astype(float)
    mask = np.abs(values - mean.filled(fill_value=np.NaN)) > std.filled(
        fill_value=np.NaN
    )

    clean = series.astype(float).copy()
    clean[mask] = np.NaN
    return clean


@register_series_method
@register_dataframe_method
def despike(df, n1=2, n2=20, block=100, keep=0):
    """
    Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`.

    """
    if isinstance(df, pd.Series):
        new_df = _despike(df, n1=n1, n2=n2, block=block, keep=keep)
    else:
        new_df = df.apply(_despike, n1=n1, n2=n2, block=block, keep=keep)
    return new_df


def _smooth(series, window_len, window):
    """Smooth the data using a window with requested size."""

    windows = {
        "flat": np.ones,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }
    data = series.values.copy()

    if window_len < 3:
        return pd.Series(data, index=series.index, name=series.name)

    if window not in list(windows.keys()):
        raise ValueError(
            """window must be one of 'flat', 'hanning',
                         'hamming', 'bartlett', 'blackman'"""
        )

    s = np.r_[
        2 * data[0] - data[window_len:1:-1],
        data,
        2 * data[-1] - data[-1:-window_len:-1],
    ]

    w = windows[window](window_len)

    data = np.convolve(w / w.sum(), s, mode="same")
    data = data[window_len - 1 : -window_len + 1]
    return pd.Series(data, index=series.index, name=series.name)


@register_series_method
@register_dataframe_method
def smooth(df, window_len=11, window="hanning"):
    """Smooth the data using a window with requested size."""
    if isinstance(df, pd.Series):
        new_df = _smooth(df, window_len=window_len, window=window)
    else:
        new_df = df.apply(_smooth, window_len=window_len, window=window)
    return new_df


def _movingaverage(series, window_size=48):
    window = np.ones(int(window_size)) / float(window_size)
    return pd.Series(np.convolve(series, window, "same"), index=series.index)


@register_series_method
@register_dataframe_method
def movingaverage(df, window_size=48):
    if isinstance(df, pd.Series):
        new_df = _movingaverage(df, window_size=window_size)
    else:
        new_df = df.apply(_movingaverage, window_size=window_size)
    return new_df
