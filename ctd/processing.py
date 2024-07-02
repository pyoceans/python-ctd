"""Processing module."""

import numpy as np
import pandas as pd
from numpy import ma
from pandas_flavor import register_dataframe_method, register_series_method

cast = pd.DataFrame | pd.Series


def _rolling_window(data: np.ndarray, block: int) -> np.ndarray:
    """http://stackoverflow.com/questions/4936620/
    Using strides for an efficient moving average filter.

    """
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = (*data.strides, data.strides[-1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


@register_series_method
@register_dataframe_method
def remove_above_water(df: cast) -> cast:
    """Remove all data above the water line."""
    return remove_up_to(df, idx=0)


@register_series_method
@register_dataframe_method
def remove_up_to(df: cast, idx: int) -> cast:
    """Remove all the data above a certain index value where index can be
    pressure or depth.
    """
    new_df = df.copy()
    return new_df[new_df.index >= idx]


@register_series_method
@register_dataframe_method
def split(df: cast) -> cast:
    """Return a tuple with down/up-cast."""
    idx = df.index.argmax() + 1
    down = df.iloc[:idx]
    # Reverse index to orient it as a CTD cast.
    up = df.iloc[idx:][::-1]
    return down, up


@register_series_method
@register_dataframe_method
def lp_filter(
    df: cast,
    sample_rate: float = 24.0,
    time_constant: float = 0.15,
) -> cast:
    """Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: 911+ systems do not require filter for temperature nor salinity.

    Examples
    --------
    >>> from pathlib import Path
    >>> import matplotlib.pyplot as plt
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> raw = ctd.from_cnv(data_path.joinpath("CTD-spiked-unfiltered.cnv.bz2"))
    >>> prc = ctd.from_cnv(data_path.joinpath("CTD-spiked-filtered.cnv.bz2"))
    >>> kw = {"sample_rate": 24.0, "time_constant": 0.15}
    >>> original = prc.index.to_numpy()
    >>> unfiltered = raw.index.to_numpy()
    >>> filtered = raw.lp_filter(**kw).index.to_numpy()
    >>> fig, ax = plt.subplots()
    >>> (l1,) = ax.plot(original, "k", label="original")
    >>> (l2,) = ax.plot(unfiltered, "r", label="unfiltered")
    >>> (l3,) = ax.plot(filtered, "g", label="filtered")
    >>> leg = ax.legend()

    Notes
    -----
    https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

    """
    from scipy import signal

    # Butter is closer to what SBE is doing with their cosine filter.
    wn = (1.0 / time_constant) / (sample_rate * 2.0)
    b, a = signal.butter(2, wn, "low")
    new_df = df.copy()
    new_df.index = signal.filtfilt(b, a, df.index.to_numpy())
    return new_df


@register_series_method
@register_dataframe_method
def press_check(df: cast) -> cast:
    """Remove pressure reversals from the index."""
    new_df = df.copy()
    press = new_df.copy().index.to_numpy()

    ref = press[0]
    inversions = np.diff(np.r_[press, press[-1]]) < 0
    mask = np.zeros_like(inversions)
    for k, p in enumerate(inversions):
        if p:
            ref = press[k]
            cut = press[k + 1 :] < ref
            mask[k + 1 :][cut] = True
    new_df[mask] = np.nan
    return new_df


def _bindata(series: pd.Series, delta: int, method: str) -> pd.Series:
    """Average the data into bins of the size `delta`."""
    start = np.ceil(series.index[0])
    stop = np.floor(series.index[-1])
    new_index = np.arange(start, stop, delta)
    binned = pd.cut(series.index, bins=new_index)
    if method == "average":
        new_series = series.groupby(binned, observed=False).mean()
        new_series.index = new_index[:-1] + delta / 2
    elif method == "interpolate":
        data = np.interp(new_index, series.index, series)
        return pd.Series(data, index=new_index, name=series.name)
    else:
        msg = f"Expected method `average` or `interpolate`, but got {method}."
        raise ValueError(
            msg,
        )
    return new_series


@register_series_method
@register_dataframe_method
def bindata(df: cast, delta: float = 1.0, method: str = "average") -> cast:
    """Bin average the index (usually pressure) to a given interval (default
    delta = 1).

    """
    if isinstance(df, pd.Series):
        new_df = _bindata(df, delta=delta, method=method)
    else:
        new_df = df.apply(_bindata, delta=delta, method=method)
    return new_df


def _despike(series: pd.Series, n1: int, n2: int, block: int) -> pd.Series:
    """Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`.

    """
    data = series.to_numpy().astype(float).copy()
    roll = _rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = np.abs(data - mean.filled(fill_value=np.nan)) > std.filled(
        fill_value=np.nan,
    )
    data[mask] = np.nan

    # Pass two recompute the mean and std without the flagged values from pass
    # one and removed the flagged data.
    roll = _rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    values = series.to_numpy().astype(float)
    mask = np.abs(values - mean.filled(fill_value=np.nan)) > std.filled(
        fill_value=np.nan,
    )

    clean = series.astype(float).copy()
    clean[mask] = np.nan
    return clean


@register_series_method
@register_dataframe_method
def despike(df: cast, n1: int = 2, n2: int = 20, block: int = 100) -> cast:
    """Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`.

    """
    if isinstance(df, pd.Series):
        new_df = _despike(df, n1=n1, n2=n2, block=block)
    else:
        new_df = df.apply(_despike, n1=n1, n2=n2, block=block)
    return new_df


def _smooth(series: pd.Series, window_len: int, window: str) -> pd.Series:
    """Smooth the data using a window with requested size."""
    windows = {
        "flat": np.ones,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }
    data = series.to_numpy().copy()

    min_window_length = 3
    if window_len < min_window_length:
        return pd.Series(data, index=series.index, name=series.name)

    if window not in list(windows.keys()):
        msg = """window must be one of 'flat', 'hanning',
                         'hamming', 'bartlett', 'blackman'"""
        raise ValueError(
            msg,
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
def smooth(df: cast, window_len: int = 11, window: str = "hanning") -> cast:
    """Smooth the data using a window with requested size."""
    if isinstance(df, pd.Series):
        new_df = _smooth(df, window_len=window_len, window=window)
    else:
        new_df = df.apply(_smooth, window_len=window_len, window=window)
    return new_df


def _movingaverage(series: pd.Series, window_size: int = 48) -> pd.Series:
    """Perform Moving Average function on a pandas series."""
    window = np.ones(int(window_size)) / float(window_size)
    return pd.Series(np.convolve(series, window, "same"), index=series.index)


@register_series_method
@register_dataframe_method
def movingaverage(df: cast, window_size: int = 48) -> cast:
    """Perform Moving Average on a DataFrame or Series.

    Inputs:
      windows_size : integer

    """
    if isinstance(df, pd.Series):
        new_df = _movingaverage(df, window_size=window_size)
    else:
        new_df = df.apply(_movingaverage, window_size=window_size)
    return new_df
