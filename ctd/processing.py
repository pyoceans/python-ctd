# -*- coding: utf-8 -*-
#
# processing.py
#
# purpose:  Functions and methods for CTD data processing.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  23-Jul-2013
# modified: Tue 23 Jul 2013 01:07:59 PM BRT
#
# obs:
#

# Scientific stack.
import gsw
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import signal
from pandas import Series, Index

from utilities import rolling_window

__all__ = ['data_conversion',
           'align',
           'despike',
           'seabird_filter',
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

    data = self.values.copy()
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
    mask = (np.abs(self.values - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    self[mask] = np.NaN
    return self


def seabird_filter(data, sample_rate=24.0, time_constant=0.15):
    """
    Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: Seabird actually uses a cosine window filter, here we use a kaiser
    window instead.
    NOTE: 911+ systems do not require filter for temperature nor salinity.

    Examples
    --------
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    >>> sample_rate, nsamples = 100., 400.
    >>> t = np.arange(nsamples) / sample_rate
    >>> x = (np.cos(2 * np.pi * 0.5 * t) +
    ...      0.2 * np.sin(2 * np.pi * 2.5 * t + 0.1) +
    ...      0.2 * np.sin(2 * np.pi * 15.3 * t) + 0.1 *
    ...      np.sin(2 * np.pi * 16.7 * t + 0.1) +
    ...      0.1 * np.sin(2 * np.pi * 23.45 * t + 0.8))
    >>> cutoff_hz = 10.0
    >>> nyq_rate = sample_rate / 2.
    >>> width = 5.0 / nyq_rate
    >>> N, beta = signal.kaiserord(60.0, width)
    >>> taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
    >>> filtered_x = seabird_filter(x, sample_rate=sample_rate,
    ...                             time_constant=1 / cutoff_hz)
    >>> fig, (ax0, ax1, ax3) = plt.subplots(nrows=3)
    >>> _ = ax0.plot(taps, 'bo-', linewidth=2)
    >>> _ = ax0.set_title('Filter Coefficients (%d taps)' % N)
    >>> ax0.grid(True)
    >>> w, h = signal.freqz(taps, worN=8000)
    >>> _ = ax1.plot((w / np.pi) * nyq_rate, np.abs(h), linewidth=2)
    >>> _ = ax1.set_xlabel('Frequency (Hz)')
    >>> _ = ax1.set_ylabel('Gain')
    >>> _ = ax1.set_title('Frequency Response')
    >>> _ = ax1.set_ylim(-0.05, 1.05)
    >>> ax1.grid(True)
    >>> # Upper inset plot.
    >>> axu = inset_axes(ax1, width="20%", height="20%", loc=4)
    >>> _ = axu.plot((w / np.pi) * nyq_rate, np.abs(h), linewidth=2)
    >>> _ = axu.set_xlim(0, 8.0)
    >>> _ = axu.set_ylim(0.9985, 1.001)
    >>> axu.grid(True)
    >>> # Lower inset plot.
    >>> _ = axl = inset_axes(ax1, width="20%", height="20%", loc=5)
    >>> _ = axl.plot((w / np.pi) * nyq_rate, np.abs(h), linewidth=2)
    >>> _ = axl.set_xlim(12.0, 20.0)
    >>> _ = axl.set_ylim(0.0, 0.0025)
    >>> axl.grid(True)
    >>> # Plot the original signal.
    >>> _ = ax3.plot(t, x)
    >>> # Plot the filtered signal.
    >>> _ = ax3.plot(t, filtered_x, 'r-')
    >>> # Plot just the "good" part of the filtered signal.  The first N-1
    >>> # samples are "corrupted" by the initial conditions.
    >>> _ = ax3.plot(t, filtered_x, 'g', linewidth=4)
    >>> _ = ax3.set_xlabel('t')
    >>> _ = ax3.grid(True)

    NOTES
    -----
    http://wiki.scipy.org/Cookbook/FIRFilter
    """

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
    Remove pressure reversal.
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
