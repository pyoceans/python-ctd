# -*- coding: utf-8 -*-


from __future__ import (absolute_import, division, print_function)

import matplotlib.pyplot as plt

import mpl_toolkits.axisartist as AA

from mpl_toolkits.axes_grid1 import host_subplot

import numpy as np
import numpy.ma as ma

from pandas import Series

from .utilities import extrap1d


def get_maxdepth(self):
    valid_last_depth = self.apply(Series.notnull).values.T
    return np.float_(self.index.values * valid_last_depth).max(axis=1)


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

    """
    from scipy.interpolate import interp1d

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

    Author
    ------
    André Palóczy Filho (paloczy@gmail.com) --  October/2012

    """

    import gsw
    from scipy.interpolate import interp1d

    h, lon, lat = list(map(np.asanyarray, (h, lon, lat)))
    # Distance in km.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    h = -gsw.z_from_p(h, lat.mean())
    Ih = interp1d(x, h, kind=kind, bounds_error=False, fill_value=h[-1])
    xm = np.arange(0, x.max() + dx, dx)
    hm = Ih(xm)

    return xm, hm


def plot(self, **kw):
    """
    Plot a CTD variable against the index (pressure or depth).
    """
    figsize = kw.pop('figsize', (5.5, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(self.values, self.index, **kw)
    ax.set_ylabel(self.index.name)
    ax.set_xlabel(self.name)
    ax.invert_yaxis()
    offset = 0.01
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
    return fig, ax


def plot_vars(self, variables=None, **kwds):
    """
    Plot CTD temperature and salinity.
    """
    fig = plt.figure(figsize=(8, 10))
    ax0 = host_subplot(111, axes_class=AA.Axes)
    ax1 = ax0.twiny()

    # Axis location.
    host_new_axis = ax0.get_grid_helper().new_fixed_axis
    ax0.axis['bottom'] = host_new_axis(loc='top', axes=ax0, offset=(0, 0))
    par_new_axis = ax1.get_grid_helper().new_fixed_axis
    ax1.axis['top'] = par_new_axis(loc='bottom', axes=ax1, offset=(0, 0))

    ax0.plot(self[variables[0]], self.index, 'r.', label='Temperature')
    ax1.plot(self[variables[1]], self.index, 'b.', label='Salinity')

    ax0.set_ylabel('Pressure [dbar]')
    ax0.set_xlabel('Temperature [\u00b0C]')
    ax1.set_xlabel('Salinity [kg g$^{-1}$]')
    ax1.invert_yaxis()

    try:
        fig.suptitle(r'Station %s profile' % self.name)
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


def plot_section(self, reverse=False, filled=False, **kw):
    import gsw

    lon, lat, data = list(map(np.asanyarray,
                              (self.lon, self.lat, self.values)))
    data = ma.masked_invalid(data)
    h = self.get_maxdepth()
    if reverse:
        lon = lon[::-1]
        lat = lat[::-1]
        data = data.T[::-1].T
        h = h[::-1]
    lon, lat = map(np.atleast_2d, (lon, lat))
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    z = self.index.values.astype(float)

    if filled:  # CAVEAT: this method cause discontinuities.
        data = data.filled(fill_value=np.nan)
        data = extrap_sec(data, x, z, w1=0.97, w2=0.03)

    # Contour key words.
    fmt = kw.pop('fmt', '%1.0f')
    extend = kw.pop('extend', 'both')
    fontsize = kw.pop('fontsize', 12)
    labelsize = kw.pop('labelsize', 11)
    cmap = kw.pop('cmap', plt.cm.rainbow)
    levels = kw.pop('levels', np.arange(np.floor(data.min()),
                    np.ceil(data.max()) + 0.5, 0.5))

    # Colorbar key words.
    pad = kw.pop('pad', 0.04)
    aspect = kw.pop('aspect', 40)
    shrink = kw.pop('shrink', 0.9)
    fraction = kw.pop('fraction', 0.05)

    # Topography mask key words.
    dx = kw.pop('dx', 1.)
    kind = kw.pop('kind', 'linear')
    linewidth = kw.pop('linewidth', 1.5)

    # Station symbols key words.
    station_marker = kw.pop('station_marker', None)
    color = kw.pop('color', 'k')
    offset = kw.pop('offset', -5)
    alpha = kw.pop('alpha', 0.5)

    # Figure.
    figsize = kw.pop('figsize', (12, 6))
    fig, ax = plt.subplots(figsize=figsize)
    xm, hm = gen_topomask(h, lon, lat, dx=dx, kind=kind)
    ax.plot(xm, hm, color='black', linewidth=linewidth, zorder=3)
    ax.fill_between(xm, hm, y2=hm.max(), color='0.9', zorder=3)

    if station_marker:
        ax.plot(x, [offset] * len(h), color=color, marker=station_marker,
                alpha=alpha, zorder=5)
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

    # Color version.
    cs = ax.contourf(x, z, data, cmap=cmap, levels=levels, alpha=1.,
                     extend=extend, zorder=2)  # manual=True
    # Colorbar.
    cb = fig.colorbar(mappable=cs, ax=ax, orientation='vertical',
                      aspect=aspect, shrink=shrink, fraction=fraction,
                      pad=pad)
    return fig, ax, cb


if __name__ == '__main__':
    import doctest
    doctest.testmod()
