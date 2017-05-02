from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
import numpy.ma as ma

from pandas import Index, Series

from .utilities import rolling_window

data_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'tests', 'data'
)


def despike(self, n1=2, n2=20, block=100, keep=0):
    """
    Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`.
    """

    data = self.values.astype(float).copy()
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(data - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    data[mask] = np.NaN

    # Pass two recompute the mean and std without the flagged values from pass
    # one and removed the flagged data.
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    values = self.values.astype(float)
    mask = (np.abs(values - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))

    clean = self.astype(float).copy()
    clean[mask] = np.NaN
    return clean


def lp_filter(data, sample_rate=24.0, time_constant=0.15):
    """
    Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: 911+ systems do not require filter for temperature nor salinity.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from ctd import DataFrame, lp_filter
    >>> name = 'CTD-spiked-unfiltered.cnv.bz2'
    >>> raw = DataFrame.from_cnv('{}/{}'.format(data_path, name),
    ...                          compression='bz2')
    >>> name = 'CTD-spiked-filtered.cnv.bz2'
    >>> prc = DataFrame.from_cnv('{}/{}'.format(data_path, name),
    ...                          compression='bz2')
    >>> kw = dict(sample_rate=24.0, time_constant=0.15)
    >>> original = prc.index.values
    >>> unfiltered = raw.index.values
    >>> filtered = lp_filter(unfiltered, **kw)
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(original, 'k', label='original')
    >>> l2, = ax.plot(unfiltered, 'r', label='unfiltered')
    >>> l3, = ax.plot(filtered, 'g', label='filtered')
    >>> leg = ax.legend()
    >>> _ = ax.axis([33564, 33648, 1034, 1035])

    NOTES
    -----
    http://wiki.scipy.org/Cookbook/FIRFilter

    """

    from scipy import signal

    if True:  # Butter is closer to what SBE is doing with their cosine filter.
        Wn = (1. / time_constant) / (sample_rate * 2.)
        b, a = signal.butter(2, Wn, 'low')
        data = signal.filtfilt(b, a, data)

    return data


def cell_thermal_mass(temperature, conductivity):
    """
    Sample interval is measured in seconds.
    Temperature in degrees.
    CTM is calculated in S/m.

    """

    alpha = 0.03  # Thermal anomaly amplitude.
    beta = 1. / 7  # Thermal anomaly time constant (1/beta).

    sample_interval = 1 / 15.
    a = 2 * alpha / (sample_interval * beta + 2)
    b = 1 - (2 * a / alpha)
    dCodT = 0.1 * (1 + 0.006 * [temperature - 20])
    dT = np.diff(temperature)
    ctm = -1.0 * b * conductivity + a * (dCodT) * dT  # [S/m]
    return ctm


def press_check(self, column='index'):
    """
    Remove pressure reversals.

    """
    data = self.copy()
    if column != 'index':
        press = data[column]
    else:
        press = data.index.values.astype(float)

    ref = press[0]
    inversions = np.diff(np.r_[press, press[-1]]) < 0
    mask = np.zeros_like(inversions)
    for k, p in enumerate(inversions):
        if p:
            ref = press[k]
            cut = press[k + 1:] < ref
            mask[k + 1:][cut] = True
    data[mask] = np.NaN
    return data


def bindata(self, delta=1., method='averaging'):
    """
    Bin average the index (usually pressure) to a given interval (default
    delta = 1).

    Note that this method does not drop NA automatically.  Therefore, one can
    check the quality of the binned data.

    """
    if method == 'averaging':
        start = np.floor(self.index[0])
        end = np.ceil(self.index[-1])
        shift = delta / 2.  # To get centered bins.
        new_index = np.arange(start, end, delta) - shift
        new_index = Index(new_index)
        newdf = self.groupby(new_index.asof).mean()
        newdf.index += shift  # Not shifted.
    else:
        newdf = self.copy()

    return newdf


def split(self):
    """
    Returns a tuple with down/up-cast.
    """
    down = self.iloc[:self.index.argmax()]
    up = self.iloc[self.index.argmax():][::-1]  # Reverse up index.
    return down, up


def movingaverage(series, window_size=48):
    window = np.ones(int(window_size)) / float(window_size)
    return Series(np.convolve(series, window, 'same'), index=series.index)


def smooth(self, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    """

    windows = dict(flat=np.ones, hanning=np.hanning, hamming=np.hamming,
                   bartlett=np.bartlett, blackman=np.blackman)
    data = self.values.copy()

    if window_len < 3:
        return Series(data, index=self.index, name=self.name)

    if window not in list(windows.keys()):
        raise ValueError("""window must be one of 'flat', 'hanning',
                         'hamming', 'bartlett', 'blackman'""")

    s = np.r_[2 * data[0] - data[window_len:1:-1], data, 2 *
              data[-1] - data[-1:-window_len:-1]]

    w = windows[window](window_len)

    data = np.convolve(w / w.sum(), s, mode='same')
    data = data[window_len - 1:-window_len + 1]
    return Series(data, index=self.index, name=self.name)


def mixed_layer_depth(CT, method='half degree'):
    if method == 'half degree':
        mask = CT[0] - CT < 0.5
    else:
        mask = np.zeros_like(CT)
    return Series(mask, index=CT.index, name='MLD')


def barrier_layer_thickness(SA, CT):
    """
    Compute the thickness of water separating the mixed surface layer from the
    thermocline.  A more precise definition would be the difference between
    mixed layer depth (MLD) calculated from temperature minus the mixed layer
    depth calculated using density.

    """
    import gsw
    sigma_theta = gsw.sigma0(SA, CT)
    mask = mixed_layer_depth(CT)
    mld = np.where(mask)[0][-1]
    sig_surface = sigma_theta[0]
    sig_bottom_mld = gsw.sigma0(SA[0], CT[mld])
    d_sig_t = sig_surface - sig_bottom_mld
    d_sig = sigma_theta - sig_bottom_mld
    mask = d_sig < d_sig_t  # Barrier layer.
    return Series(mask, index=SA.index, name='BLT')


def derive_cnv(self):
    """
    Compute SP, SA, CT, z, and GP from a cnv pre-processed cast.

    """
    import gsw
    cast = self.copy()
    p = cast.index.values.astype(float)
    cast['SP'] = gsw.SP_from_C(cast['c0S/m'].values * 10.,
                               cast['t090C'].values, p)
    cast['SA'] = gsw.SA_from_SP(cast['SP'].values, p, self.lon, self.lat)
    cast['SR'] = gsw.SR_from_SP(cast['SP'].values)
    cast['CT'] = gsw.CT_from_t(cast['SA'].values, cast['t090C'].values, p)
    cast['z'] = -gsw.z_from_p(p, self.lat)
    cast['sigma0_CT'] = gsw.sigma0(cast['SA'].values, cast['CT'].values)
    return cast


if __name__ == '__main__':
    import doctest
    doctest.testmod()
