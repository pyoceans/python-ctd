from __future__ import absolute_import, division, print_function

import os,re
import warnings
from datetime import datetime

import numpy as np

from pandas import DataFrame
from pandas import read_table, read_fwf

from .utilities import basename, normalize_names, read_file

data_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'tests', 'data'
)


def asof(self, label):
    """pandas index workaround."""
    if label not in self:
        loc = self.searchsorted(label, side='left')
        if loc > 0:
            return self[loc - 1]
        else:
            return np.nan
    return label


class CTD(DataFrame):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        name=None,
        time=None,
        longitude=None,
        latitude=None,
        header=None,
        serial=None,
        config=None,
        dtype=None,
        copy=False
    ):
        super(CTD, self).__init__(
            data=data,
            index=index,
            columns=columns,
            dtype=dtype,
            copy=copy
        )
        self.time = time
        self.longitude = longitude
        self.latitude = latitude
        self.header = header
        self.serial = serial
        self.config = config
        self.name = name

    def __reduce__(self):
        return self.__class__, (
            DataFrame(self),  # NOTE Using that type(data)==DataFrame and the
                              # the rest of the arguments of DataFrame.__init__
                              # to defaults, the constructors acts as a
                              # copy constructor.
            None,
            None,
            self.time,
            self.longitude,
            self.latitude,
            self.header,
            self.serial,
            self.config,
            self.name,
            None,
            False,
        )


def remove_above_water(cast):
    return cast[cast.index >= 0]


def from_edf(fname, compression=None, below_water=False, lon=None,
             lat=None):
    """
    DataFrame constructor to open XBT EDF ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_edf('{}/{}'.format(data_path, 'XBT.EDF.gz'),
    ...                           compression='gzip')
    >>> fig, ax = cast['temperature'].plot()
    >>> _ = ax.axis([20, 24, 19, 0])
    >>> ax.grid(True)
    """
    f = read_file(fname, compression=compression)
    header, names = [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if line.startswith('Serial Number'):
            serial = line.strip().split(':')[1].strip()
        elif line.startswith('Latitude'):
            if not lat:
                try:
                    hemisphere = line[-1]
                    lat = line.strip(hemisphere).split(':')[1].strip()
                    lat = np.float_(lat.split())
                    if hemisphere == 'S':
                        lat = -(lat[0] + lat[1] / 60.)
                    elif hemisphere == 'N':
                        lat = lat[0] + lat[1] / 60.
                except (IndexError, ValueError) as e:
                    msg = 'Ill formed or not present latitude in the header. '
                    msg += 'Try specifying one with the keyword `lat=`.'
                    raise ValueError('%s\n%s' % (msg, e))
        elif line.startswith('Longitude'):
            if not lon:
                try:
                    hemisphere = line[-1]
                    lon = line.strip(hemisphere).split(':')[1].strip()
                    lon = np.float_(lon.split())
                    if hemisphere == 'W':
                        lon = -(lon[0] + lon[1] / 60.)
                    elif hemisphere == 'E':
                        lon = lon[0] + lon[1] / 60.
                except (IndexError, ValueError) as e:
                    msg = 'Ill formed or not present Longitude in the header.'
                    msg += 'Try specifying one with the keyword `lon=`.'
                    raise ValueError('%s\n%s' % (msg, e))
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
                      skiprows=skiprows, delim_whitespace=True)
    f.close()

    cast.set_index('depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    name = basename(fname)[1]
    if below_water:
        cast = remove_above_water(cast)
    return CTD(cast, longitude=float(lon), latitude=float(lat),
               serial=serial, name=name, header=header)


def from_cnv(fname, compression=None, below_water=False, time=None,
             lon=None, lat=None):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_cnv('{}/{}'.format(data_path,
    ...                                          'CTD_big.cnv.bz2'),
    ...                           compression='bz2')
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['t090C'].plot()
    >>> ax.grid(True)
    """

    f = read_file(fname, compression=compression)
    header, config, names = [], [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if '# name' in line:  # Get columns names.
            name, unit = line.split('=')[1].split(':')
            name, unit = list(map(normalize_names, (name, unit)))
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
                raise ValueError('Latitude not recognized.')
        if 'NMEA Longitude' in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split('=')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError('Latitude not recognized.')
        if 'NMEA UTC (Time)' in line:
            time = line.split('=')[-1].strip()
            # Should use some fuzzy datetime parser to make this more robust.
            time = datetime.strptime(time, '%b %d %Y %H:%M:%S')
        if line == '*END*':  # Get end of header.
            skiprows = k + 1
            break

    f.seek(0)
    cast = read_table(f, header=None, index_col=None, names=names,
                      skiprows=skiprows, delim_whitespace=True)
    f.close()

    key_set = False
    prkeys = ['prDM', 'prdM', 'pr']
    for prkey in prkeys:
        try:
            cast.set_index(prkey, drop=True, inplace=True)
            key_set = True
        except KeyError:
            continue
    if not key_set:
        msg = 'Could not find pressure field (supported names are {}).'
        raise KeyError(msg.format(prkeys))
    cast.index.name = 'Pressure [dbar]'

    name = basename(fname)[0]

    dtypes = {
        'bpos': int,
        'pumps': bool,
        'flag': bool
    }
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = cast[column].astype(float)
            except ValueError:
                warnings.warn('Could not convert %s to float.' % column)
    if below_water:
        cast = remove_above_water(cast)
    return CTD(
        cast,
        time=time,
        longitude=lon,
        latitude=lat,
        name=name,
        header=header,
        config=config
        )


def from_btl(fname, compression=None, below_water=False, lon=None,
             lat=None):
    """
    DataFrame constructor to open Seabird CTD BTL-ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> bottles = DataFrame.from_btl(filepath)
    """

    f = read_file(fname, compression=compression)
    header, config, names = [], [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        ''' #bottle files can't get variable names like this
        if '# name' in line:  # Get columns names.
            name, unit = line.split('=')[1].split(':')
            name, unit = list(map(normalize_names, (name, unit)))
            names.append(name)
        '''
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
                raise ValueError('Latitude not recognized.')
        if 'NMEA Longitude' in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split('=')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError('Latitude not recognized.')
        # There is no *END* like in a .cnv file, skip two after header info.
        if not (line.startswith('*') | line.startswith('#') ):
            # Fix commonly occurring problem when Sbeox.* exists in the file
            # the name is concatenated to previous parameter
            # example:
            #   CStarAt0Sbeox0Mm/Kg to CStarAt0 Sbeox0Mm/Kg (really two different params)
            line = re.sub(r'(\S)Sbeox', '\\1 Sbeox', line)

            names = line.split()
            skiprows = k + 2
            break

    f.seek(0)
    # Capture stat names column.
    names.append('Statistic')

    df = read_fwf(f, header=None, index_col=False, names=names,parse_dates=False,
                      skiprows=skiprows)
    f.close()

    # At this point the data frame is not correctly lined up (multiple rows for avg, std, min, max or
    # just avg, std, etc).
    # Also needs date,time,and bottle number to be converted to one per line.

    # Add new column and put values in it.
    df.insert(2, 'Time', df['Date'])

    # Get row types, see what you have: avg, std, min, max or just avg, std.
    rowtypes = df[df.columns[-1]].unique()
    # Get dates which occur on second line of each bottle.
    dates = df.iloc[::len(rowtypes), 1]
    # Get times which occur on second line of each bottle.
    times = df.iloc[1::len(rowtypes), 1]

    # Add times to first rows.
    df['Time'].iloc[::4] = times.values

    # Remove time from date column.
    df['Date'].iloc[1::4] = dates.values

    # Fill missing rows.
    df['Bottle'] = df['Bottle'].fillna(method='ffill')
    df['Date'] = df['Date'].fillna(method='ffill')
    df['Time'] = df['Time'].fillna(method='ffill')

    df['Statistic'] = df['Statistic'].str.replace(r'\(|\)', '')  # (avg) to avg
    # avg_df = df.iloc[::4, :]
    # cast = avg_df  # Return just avg as cast.

    cast = df

    # Not setting pressure as index.

    name = basename(fname)[0]

    dtypes = dict(bpos=int, pumps=bool, flag=bool)
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = cast[column].astype(float)
            except ValueError:
                if column not in ('Bottle', 'Date', 'Time', 'Statistic'):
                    warnings.warn('Could not convert %s to float.' % column)
    if below_water:
        cast = remove_above_water(cast)

    cast['Bottle'] = cast['Bottle'].astype(int)
    cast['Scan'] = cast['Scan'].astype(int)
    return CTD(cast, longitude=lon, latitude=lat, name=name, header=header,
               config=config)


def from_fsi(fname, compression=None, skiprows=9, below_water=False,
             lon=None, lat=None):
    """
    DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_fsi('{}/{}'.format(data_path, 'FSI.txt.gz'),
    ...                           compression='gzip')
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['TEMP'].plot()
    >>> ax.grid(True)

    """
    f = read_file(fname, compression=compression)
    cast = read_table(f, header='infer', index_col=None, skiprows=skiprows,
                      dtype=float, delim_whitespace=True)
    f.close()

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
    >>> fname = '{}/{}'.format(data_path, 'CTD/g01l01s01.ros')
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
