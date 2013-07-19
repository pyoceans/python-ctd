# -*- coding: utf-8 -*-
#
# ctd.py
#
# purpose:  Some classes and functions to work with CTD data.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2012
# modified: Fri 19 Jul 2013 05:46:11 PM BRT
#
# obs: New constructors and methods for pandas DataFrame and Series.
#


# Standard library.
import os
import bz2
import gzip
import zipfile
from cStringIO import StringIO
from xml.etree import cElementTree as etree

# Scientific stack.
import gsw
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA

from scipy import signal
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot
from pandas import DataFrame, Series, Index, read_table

__all__ = ['rosette_summary',
           'movingaverage',
           'seabird_filter',
           'pmel_inversion_check',
           'cell_theramll_mass',
           'mixed_layer_depth',
           'barrier_layer_thickness',
           'extrap_sec',
           'gen_topomask',
           'from_fsi',
           'from_edf',
           'from_cnv']


# Utilities.
def header(xml):
    return etree(xml)


def basename(fname):
    """Return filename without path.
    Examples
    ========
    >>> fname = '../test/data/FSI.txt.zip'
    >>> basename(fname)
    ('../test/data', 'FSI.txt', '.zip')
    """
    path, name = os.path.split(fname)
    name, ext = os.path.splitext(name)
    return path, name, ext


def rolling_window(data, block):
    """
    http://stackoverflow.com/questions/4936620/
    Using strides for an efficient moving average filter.
    """
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def extrap1d(interpolator):
    """
    http://stackoverflow.com/questions/2745329/
    How to make scipy.interpolate give an extrapolated result beyond the
    input range.
    """
    xs, ys = interpolator.x, interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return (ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) /
                    (xs[-1] - xs[-2]))
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def normalize_names(name):
    name = name.strip()
    name = name.lower()
    name = name.strip('*')
    return name


def read_file(fname, compression=None):
    if compression == 'gzip':
        lines = gzip.open(fname)
    elif compression == 'bz2':
        lines = bz2.BZ2File(fname)
    elif compression == 'zip':
        zfile = zipfile.ZipFile(fname)
        # Zip format might contain more than one file in the archive (similar
        # to tar), here we assume that there is just one file per zipfile.
        name = zfile.namelist()[0]
        lines = StringIO(zfile.read(name))
    else:
        lines = open(fname)
    return lines


# Pre-processing.
def rosette_summary(rosfile):
    """
    Make a BTL (bottle) file from a ROS (bottle log) file.

    More control for the averaging process and at which step we want to
    perform this averaging eliminating the need to read the data into SBE
    Software again after pre-processing.
    NOTE: Do not run LoopEdit on the upcast!
    """
    ros = DataFrame.from_cnv(rosfile)
    ros['pressure'] = ros.index.values.astype(float)
    ros['nbf'] = ros['nbf'].astype(int)  # Make bottle number as string.
    ros.set_index('nbf', drop=True, inplace=True, verify_integrity=False)
    return ros


def movingaverage(series, window_size=48):
    window = np.ones(int(window_size)) / float(window_size)
    return Series(np.convolve(series, window, 'same'), index=series.index)


def seabird_filter(data, sample_rate=24.0, time_constant=0.15):
    """
    Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: Seabird actually uses a cosine window filter, here we use a kaiser
    window instead.
    NOTE: 911+ systems do not require filter for temperature nor salinity.
    """

    nyq_rate = sample_rate / 2.0
    width = 5.0 / nyq_rate  # 5 Hz transition rate.
    ripple_db = 60.0  # Attenuation at the stop band.
    N, beta = signal.kaiserord(ripple_db, width)

    cutoff_hz = (1. / time_constant)  # Cutoff frequency at 0.15 s.
    taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
    data = signal.filtfilt(taps, [1.0], data)
    return data


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


def cell_theramll_mass(temperature, conductivity):
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


# Processing.
def mixed_layer_depth(CT, method='half degree'):
    if method == 'half degree':
        mask = CT[0] - CT < 0.5
    else:
        mask = np.zeros_like(CT)
    return Series(mask, index=CT.index, name='MLD')


def barrier_layer_thickness(SA, CT):
    """
    compute the thickness of water separating the mixed surface layer from the
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


def extrap_sec(data, dist, depth, w1=1., w2=0):
    """
    Extrapolates `data` to zones where the shallow stations are shadowed by
    the deep stations.  The shadow region usually cannot be extrapolates via
    linear interpolation.

    The extrapolation is applied using the gradients of the `data` at a certain
    level.

    Parameters
    ----------
    data : array_like
          Data to be extrapolated
    dist : array_like
           Stations distance
    fd : float
         Decay factor [0-1]


    Returns
    -------
    Sec_extrap : array_like
                 Extrapolated variable

    Examples
    --------
    >>> import data, dist, z
    >>> Sec_extrap = extrap_sec(data, dist, z, fd=1.)
    """
    new_data1 = []
    for row in data:
        mask = ~np.isnan(row)
        if mask.any():
            y = row[mask]
            if y.size == 1:
                row = np.repeat(y, len(mask))
            else:
                x = dist[mask]
                f_i = interp1d(x, y)
                f_x = extrap1d(f_i)
                row = f_x(dist)
        new_data1.append(row)

    new_data2 = []
    for col in data.T:
        mask = ~np.isnan(col)
        if mask.any():
            y = col[mask]
            if y.size == 1:
                col = np.repeat(y, len(mask))
            else:
                z = depth[mask]
                f_i = interp1d(z, y)
                f_z = extrap1d(f_i)
                col = f_z(depth)
        new_data2.append(col)

    new_data = np.array(new_data1) * w1 + np.array(new_data2).T * w2
    return new_data


def gen_topomask(h, lon, lat, dx=1., kind='linear', plot=False):
    """
    Generates a topography mask from an oceanographic transect taking the
    deepest CTD scan as the depth of each station.

    Inputs
    ------
    h : array
        Pressure of the deepest CTD scan for each station [dbar].
    lons : array
           Longitude of each station [decimal degrees east].
    lat : Latitude of each station. [decimal degrees north].
    dx : float
         Horizontal resolution of the output arrays [km].
    kind : string, optional
           Type of the interpolation to be performed.
           See scipy.interpolate.interp1d documentation for details.
    plot : bool
           Whether to plot mask for visualization.

    Outputs
    -------
    xm : array
         Horizontal distances [km].
    hm : array
         Local depth [m].

    Examples
    --------
    >>> import gsw, df
    >>> h = df.get_maxdepth()
    >>> # TODO: method to output distance.
    >>> x = np.append(0, np.cumsum(gsw.distance(df.lon, df.lat)[0] / 1e3))
    >>> xm, hm = gen_topomask(h, df.lon, df.lat, dx=1., kind='linear')
    >>> fig, ax = plt.subplots()
    >>> ax.plot(xm, hm, 'k', linewidth=1.5)
    >>> ax.plot(x, h, 'ro')
    >>> ax.set_xlabel('Distance [km]')
    >>> ax.set_ylabel('Depth [m]')
    >>> ax.grid(True)
    >>> plt.show()

    Author
    ------
    André Palóczy Filho (paloczy@gmail.com) --  October/2012
    """

    h, lon, lat = map(np.asanyarray, (h, lon, lat))
    # Distance in km.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    h = -gsw.z_from_p(h, lat.mean())
    Ih = interp1d(x, h, kind=kind, bounds_error=False, fill_value=h[-1])
    xm = np.arange(0, x.max() + dx, dx)
    hm = Ih(xm)

    return xm, hm


# Index methods.
def asof(self, label):
    """FIXME: pandas index workaround."""
    if label not in self:
        loc = self.searchsorted(label, side='left')
        if loc > 0:
            return self[loc - 1]
        else:
            return np.nan
    return label


def split(self):
    """
    Returns a tuple with down/up-cast.
    """
    down = self.ix[:self.index.argmax()]
    up = self.ix[self.index.argmax():][::-1]  # Reverse up index.
    return down, up


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


# Series methods.
def plot(self, **kwds):
    """
    Plot a CTD variable against the index (pressure or depth).
    """
    fig, ax = plt.subplots()
    ax.plot(self.values, self.index, **kwds)
    ax.set_ylabel(self.index.name)
    ax.set_xlabel(self.name)
    ax.invert_yaxis()
    offset = 0.01
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
    return fig, ax


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


def get_maxdepth(self):
    valid_last_depth = self.apply(Series.notnull).values.T
    return np.float_(self.index * valid_last_depth).max(axis=1)


# DataFrame methods.
def plot_vars(self, variables=None, **kwds):
    """
    Plot CTD temperature and salinity.
    """
    fig = plt.figure(figsize=(8, 10))
    ax0 = host_subplot(111, axes_class=AA.Axes)
    ax1 = ax0.twiny()

    # Axis location.
    host_new_axis = ax0.get_grid_helper().new_fixed_axis
    ax0.axis["bottom"] = host_new_axis(loc="top", axes=ax0, offset=(0, 0))
    par_new_axis = ax1.get_grid_helper().new_fixed_axis
    ax1.axis["top"] = par_new_axis(loc="bottom", axes=ax1, offset=(0, 0))

    ax0.plot(self[variables[0]], self.index, 'r.', label='Temperature')
    ax1.plot(self[variables[1]], self.index, 'b.', label='Salinity')

    ax0.set_ylabel("Pressure [dbar]")
    ax0.set_xlabel(u"Temperature [\u00b0C]")
    ax1.set_xlabel("Salinity [kg g$^{-1}$]")
    ax1.invert_yaxis()

    try:  # FIXME with metadata.
        fig.suptitle(r"Station %s profile" % self.name)
    except AttributeError:
        pass

    ax0.legend(shadow=True, fancybox=True,
               numpoints=1, loc='lower right')

    offset = 0.01
    x1, x2 = ax0.get_xlim()[0] - offset, ax0.get_xlim()[1] + offset
    ax0.set_xlim(x1, x2)

    offset = 0.01
    x1, x2 = ax1.get_xlim()[0] - offset, ax1.get_xlim()[1] + offset
    ax1.set_xlim(x1, x2)

    return fig, (ax0, ax1)


def plot_section(self, inverse=False, filled=False, **kw):
    if inverse:
        lon = self.lon[::-1].copy()
        lat = self.lat[::-1].copy()
        data = self.T[::-1].T.copy()
    else:
        lon = self.lon.copy()
        lat = self.lat.copy()
        data = self.copy()
    # Contour key words.
    fmt = kw.pop('fmt', '%1.0f')
    extend = kw.pop('extend', 'both')
    fontsize = kw.pop('fontsize', 12)
    labelsize = kw.pop('labelsize', 11)
    cmap = kw.pop('cmap', plt.cm.rainbow)
    levels = kw.pop('levels', np.arange(np.floor(data.min().min()),
                    np.ceil(data.max().max()) + 0.5, 0.5))

    # Colorbar key words.
    pad = kw.pop('pad', 0.04)
    aspect = kw.pop('aspect', 40)
    shrink = kw.pop('shrink', 0.9)
    fraction = kw.pop('fraction', 0.05)

    # Topography mask key words.
    dx = kw.pop('dx', 1.)
    kind = kw.pop('kind', 'linear')

    # Station symbols key words.
    color = kw.pop('color', 'k')
    offset = kw.pop('offset', -5)
    linewidth = kw.pop('linewidth', 1.5)

    # Get data for plotting.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    z = np.float_(data.index.values)
    h = data.get_maxdepth()
    data = ma.masked_invalid(data.values)
    if filled:
        # FIXME: Cause discontinuities.
        data = data.filled(fill_value=np.nan)
        data = extrap_sec(data, x, z, w1=0.97, w2=0.03)

    xm, hm = gen_topomask(h, lon, lat, dx=dx, kind=kind)

    # Figure.
    fig, ax = plt.subplots()
    ax.plot(xm, hm, color='black', linewidth=linewidth, zorder=3)
    ax.fill_between(xm, hm, y2=hm.max(), color='0.9', zorder=3)

    ax.plot(x, [offset] * len(h), color=color, marker='v',
            alpha=0.5, zorder=5)
    ax.set_xlabel('Cross-shore distance [km]', fontsize=fontsize)
    ax.set_ylabel('Depth [m]', fontsize=fontsize)
    ax.set_ylim(offset, hm.max())
    ax.invert_yaxis()

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.xaxis.set_tick_params(tickdir='out', labelsize=labelsize, pad=1)
    ax.yaxis.set_tick_params(tickdir='out', labelsize=labelsize, pad=1)

    if False:  # TODO: +/- Black-and-White version.
        cs = ax.contour(x, z, data, colors='grey', levels=levels,
                        extend=extend, linewidths=1., alpha=1., zorder=2)
        ax.clabel(cs, fontsize=8, colors='grey', fmt=fmt, zorder=1)
        cb = None
    if True:  # Color version.
        cs = ax.contourf(x, z, data, cmap=cmap, levels=levels, alpha=1.,
                         extend=extend, zorder=2)  # manual=True
        # Colorbar.
        cb = fig.colorbar(mappable=cs, ax=ax, orientation='vertical',
                          aspect=aspect, shrink=shrink, fraction=fraction,
                          pad=pad)
    return fig, ax, cb


# Constructors.
@classmethod
def from_fsi(cls, fname, compression=None, skiprows=9):
    """
    DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.
    """
    f = read_file(fname, compression=compression)
    cast = read_table(f, header='infer', index_col=None, dtype=float,
                      skiprows=skiprows, delim_whitespace=True)

    cast.set_index('PRES', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'
    cast.name = basename(fname)[0]
    return cast


@classmethod
def from_edf(cls, fname, compression=None):
    """
    DataFrame constructor to open XBT EDF ASCII format.
    """
    f = read_file(fname, compression=compression)
    header, names = [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if line.startswith('Serial Number'):
            serial = line.strip().split(':')[1].strip()
        elif line.startswith('Latitude'):
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split(':')[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == 'S':
                lat = -(lat[0] + lat[1] / 60.)
            elif hemisphere == 'N':
                lat = lat[0] + lat[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        elif line.startswith('Longitude'):
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split(':')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError("Longitude not recognized.")
        else:
            header.append(line)
            if line.startswith('Field'):
                col, unit = [l.strip().lower() for l in line.split(':')]
                names.append(unit.split()[0])
        if line == '// Data':
            skiprows = k + 1
            break

    f.seek(0)
    cast = read_table(f, header=None, index_col=None, names=names,
                      skiprows=skiprows, dtype=np.float_,
                      delim_whitespace=True)
    f.close()

    cast.set_index('depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    # FIXME: Try metadata class.
    cast.lon = lon
    cast.lat = lat
    cast.serial = serial
    cast.header = header
    cast.name = basename(fname)[1]

    return cast


@classmethod
def from_cnv(cls, fname, compression=None, blfile=None):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.
    """

    f = read_file(fname, compression=compression)
    header, config, names = [], [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if '# name' in line:  # Get columns names.
            name, unit = line.split('=')[1].split(':')
            name, unit = map(normalize_names, (name, unit))
            names.append(name)
        if line.startswith('*'):  # Get header.
            header.append(line)
        if line.startswith('#'):  # Get configuration file.
            config.append(line)
        if 'NMEA Latitude' in line:
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split('=')[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == 'S':
                lat = -(lat[0] + lat[1] / 60.)
            elif hemisphere == 'N':
                lat = lat[0] + lat[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        if 'NMEA Longitude' in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split('=')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        if line == '*END*':  # Get end of header.
            skiprows = k + 1
            break

    f.seek(0)
    cast = read_table(f, header=None, index_col=None, names=names,
                      skiprows=skiprows, dtype=np.float_,
                      delim_whitespace=True)
    f.close()

    cast.set_index('prdm', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'

    cast.lon = lon
    cast.lat = lat
    cast.header = header
    cast.config = config
    cast.name = basename(fname)[0]
    if 'pumps' in cast.columns:
        cast['pumps'] = np.bool_(cast['pumps'])
    if 'flag' in cast.columns:
        cast['flag'] = np.bool_(cast['flag'])
    return cast


# Attach methods.
Index.asof = asof

Series.plot = plot
Series.split = split
Series.smooth = smooth
Series.despike = despike
Series.bindata = bindata
Series.press_check = press_check

DataFrame.split = split
DataFrame.from_cnv = from_cnv
DataFrame.from_edf = from_edf
DataFrame.from_fsi = from_fsi
DataFrame.plot_vars = plot_vars
DataFrame.press_check = press_check
DataFrame.get_maxdepth = get_maxdepth
DataFrame.plot_section = plot_section

if __name__ == '__main__':
    import doctest
    doctest.testmod()
