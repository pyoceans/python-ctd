# -*- coding: utf-8 -*-
#
# ctd.py
#
# purpose:  Some classes and functions to work with CTD data.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2012
# modified: Fri 19 Jul 2013 06:24:21 PM BRT
#
# obs: New constructors and methods for pandas DataFrame and Series.
#


# Scientific stack.
import numpy as np
from pandas import read_table

from utilities import read_file, basename, normalize_names

__all__ = ['asof',
           'from_edf',
           'from_cnv',
           'from_fsi',
           'rosette_summary']


def asof(self, label):
    """FIXME: pandas index workaround."""
    if label not in self:
        loc = self.searchsorted(label, side='left')
        if loc > 0:
            return self[loc - 1]
        else:
            return np.nan
    return label


def remove_above_water(cast):
    return cast[cast.index >= 0]


def from_edf(fname, compression=None, below_water=False, lon=None,
             lat=None):
    """
    DataFrame constructor to open XBT EDF ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_edf('../test/data/XBT.EDF.gz',
    ...                           compression='gzip')
    >>> fig, ax = cast['temperature'].plot()
    >>> ax.axis([20, 24, 19, 0])
    >>> ax.grid(True)
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
    cast = read_table(f, header=None, index_col=None, names=names, dtype=float,
                      skiprows=skiprows, delim_whitespace=True)
    f.close()

    cast.set_index('depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    # FIXME: Try metadata class.
    cast.lon = lon
    cast.lat = lat
    cast.serial = serial
    cast.header = header
    cast.name = basename(fname)[1]
    if below_water:
        cast = remove_above_water(cast)
    return cast


def from_cnv(fname, compression=None, below_water=False, lon=None,
             lat=None):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_cnv('../test/data/CTD_big.cnv.bz2',
    ...                           compression='bz2')
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['t090c'].plot()
    >>> ax.grid(True)
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
                      skiprows=skiprows, delim_whitespace=True)
    f.close()

    cast.set_index('prDM', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'

    cast.lon = lon
    cast.lat = lat
    cast.header = header
    cast.config = config
    cast.name = basename(fname)[0]

    dtypes = dict(bpos=int, pumps=bool, flag=bool)
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = np.float_(cast[column].astype(float))
            except ValueError:
                print('Could not convert %s to float.' % column)
    if below_water:
        cast = remove_above_water(cast)
    return cast


def from_fsi(fname, compression=None, skiprows=9, below_water=False,
             lon=None, lat=None):
    """
    DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_fsi('../test/data/FSI.txt.zip',
    ...                           compression='zip')
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['TEMP'].plot()
    >>> ax.grid(True)
    """
    f = read_file(fname, compression=compression)
    cast = read_table(f, header='infer', index_col=None, skiprows=skiprows,
                      dtype=float, delim_whitespace=True)

    cast.set_index('PRES', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'
    cast.name = basename(fname)[0]
    if below_water:
        cast = remove_above_water(cast)
    return cast


def rosette_summary(fname):
    """
    Make a BTL (bottle) file from a ROS (bottle log) file.

    More control for the averaging process and at which step we want to
    perform this averaging eliminating the need to read the data into SBE
    Software again after pre-processing.
    NOTE: Do not run LoopEdit on the upcast!

    Examples
    --------
    >>> fname = '../test/data/CTD/g01l01s01.ros'
    >>> ros = rosette_summary(fname)
    >>> ros = ros.groupby(ros.index).mean()
    >>> np.int_(ros.pressure.values)
    array([835, 806, 705, 604, 503, 404, 303, 201, 151, 100,  51,   1])
    """
    ros = from_cnv(fname)
    ros['pressure'] = ros.index.values.astype(float)
    ros['nbf'] = ros['nbf'].astype(int)
    ros.set_index('nbf', drop=True, inplace=True, verify_integrity=False)
    return ros

if __name__ == '__main__':
    import doctest
    doctest.testmod()
