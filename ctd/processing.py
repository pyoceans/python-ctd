# -*- coding: utf-8 -*-
#
# processing.py
#
# purpose:  Functions and methods for CTD data processing.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  23-Jul-2013
# modified: Wed 24 Jul 2013 08:17:22 PM BRT
#
# obs:
#

# Scientific stack.
import gsw
import numpy as np
import numpy.ma as ma

from scipy import signal
from pandas import Series, Index

from utilities import rolling_window

__all__ = ['data_conversion',  # TODO: Add as a constructor.
           'align',
           'despike',
           'lp_filter',
           'cell_thermal_mass',
           'press_check',  # TODO: Loop edit + velocity_check
           'bindata',
           'split',
           'pmel_inversion_check',
           'smooth',
           'mixed_layer_depth',
           'barrier_layer_thickness']


# Pre-processing.
def data_conversion(hexfile):
    """TODO: Read SBE hexadecimal file (Option for from_cnv)."""
    pass


def align(conductivity):
    """TODO: Align conductivity and temperature."""
    pass


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
    mask = (np.abs(self.values.astype(float) - mean.filled(fill_value=np.NaN))
            > std.filled(fill_value=np.NaN))

    clean = self.astype(float).copy()
    clean[mask] = np.NaN
    return clean


def lp_filter(data, sample_rate=24.0, time_constant=0.15):
    """
    Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: Seabird actually uses a cosine window filter, here we use a kaiser
    window instead.
    NOTE: 911+ systems do not require filter for temperature nor salinity.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from ctd import DataFrame, lp_filter
    >>> raw = DataFrame.from_cnv('../test/data/CTD-spiked-unfiltered.cnv.bz2',
    ...                          compression='bz2')
    >>> prc = DataFrame.from_cnv('../test/data/CTD-spiked-filtered.cnv.bz2',
    ...                          compression='bz2')
    >>> kw = dict(sample_rate=24.0, time_constant=0.15)
    >>> original = prc.index.values
    >>> unfiltered = raw.index.values
    >>> filtered = lp_filter(unfiltered, **kw)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(original, 'k', label='original')
    >>> ax.plot(unfiltered, 'r', label='unfiltered')
    >>> ax.plot(filtered, 'g', label='filtered')
    >>> ax.legend()
    >>> ax.axis([33564, 33648, 1034, 1035])
    >>> plt.show()

    NOTES
    -----
    http://wiki.scipy.org/Cookbook/FIRFilter
    """

    # Butter is closer to what SBE is doing with their cosine filter.
    if True:
        Wn = (1. / time_constant) / (sample_rate * 2.)
        b, a = signal.butter(2, Wn, 'low')
        data = signal.filtfilt(b, a, data)

    if False:  # Kaiser.
        nyq_rate = sample_rate / 2.0
        width = 5.0 / nyq_rate  # 5 Hz transition rate.
        ripple_db = 60.0  # Attenuation at the stop band.
        N, beta = signal.kaiserord(ripple_db, width)

        cutoff_hz = (1. / time_constant)  # Cutoff frequency at 0.15 s.
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
        data = signal.filtfilt(taps, [1.0], data)
    return data


def cell_thermal_mass(temperature, conductivity):
    """
    FIXME: UNFINISHED!
    Sample interval is measured in seconds.
    Temperature in °C
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
    if column is not 'index':
        press = data[column]
    else:
        press = data.index.astype(float)

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
    elif method == 'interpolate':
        # TODO:
        newdf = self.copy()
    else:
        newdf = self.copy()

    return newdf


def split(self):
    """
    Returns a tuple with down/up-cast.
    """
    down = self.ix[:self.index.argmax()]
    up = self.ix[self.index.argmax():][::-1]  # Reverse up index.
    return down, up


# Pos-processing.
def pmel_inversion_check():
    """
    FIXME: UNFINISHED!.
    Additional clean-up and flagging of data after the SBE Processing.
    Look for inversions in the processed, binned via computing the centered
    square of the buoyancy frequency, N2, for each bin and linearly
    interpolating temperature, conductivity, and oxygen over those records
    where N2 ≤ -1 x 10-5 s-2, where there appear to be density inversions.

    NOTE: While these could be actual inversions in the CTD records, it is much
    more likely that shed wakes cause these anomalies.  Records that fail the
    density inversion criteria in the top 20 meters are retained, but flagged
    as questionable.

    FIXME: The codes also manually remove spikes or glitches from profiles as
    necessary, and linearly interpolate over them.
    """

    # TODO
    pass


def smooth(self, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    """

    windows = dict(flat=np.ones, hanning=np.hanning, hamming=np.hamming,
                   bartlett=np.bartlett, blackman=np.blackman)
    data = self.values.copy()

    if window_len < 3:
        return Series(data, index=self.index, name=self.name)

    if not window in windows.keys():
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
    sigma_theta = gsw.sigma0_CT_exact(SA, CT)
    mask = mixed_layer_depth(CT)
    mld = np.where(mask)[0][-1]
    sig_surface = sigma_theta[0]
    sig_bottom_mld = gsw.sigma0_CT_exact(SA[0], CT[mld])
    d_sig_t = sig_surface - sig_bottom_mld
    d_sig = sigma_theta - sig_bottom_mld
    mask = d_sig < d_sig_t  # Barrier layer.
    return Series(mask, index=SA.index, name='BLT')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
