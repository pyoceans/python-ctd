from __future__ import absolute_import, division, print_function

import re
import warnings
from datetime import datetime
import linecache
from netCDF4 import Dataset, date2num,num2date
import numpy as np

import pandas as pd
if "time_of_reset" not in pd.DataFrame._metadata:
    pd.DataFrame._metadata.append("time_of_reset")


from .utilities import (
    Path,
    basename,
    normalize_names,
    read_file,
)

data_path = Path(__file__).parents[1].joinpath('tests', 'data')


def _parse_seabird(lines, ftype='cnv'):
    # Initialize variables.
    lon = lat = time = None
    skiprows = 0

    metadata = {}
    header, config, names = [], [], []
    for k, line in enumerate(lines):
        line = line.strip()

        # Only cnv has columns names, for bottle files we will use the variable row.
        if ftype == 'cnv':
            if '# name' in line:
                name, unit = line.split('=')[1].split(':')
                name, unit = list(map(normalize_names, (name, unit)))
                names.append(name)

        # Seabird headers starts with *.
        if line.startswith('*'):
            header.append(line)

        # Seabird configuration starts with #.
        if line.startswith('#'):
            config.append(line)

        # NMEA position and time.
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

        # cnv file header ends with *END* while
        if ftype == 'cnv':
            if line == '*END*':
                skiprows = k + 1
                break
        else:  # btl.
            # There is no *END* like in a .cnv file, skip two after header info.
            if not (line.startswith('*') | line.startswith('#')):
                # Fix commonly occurring problem when Sbeox.* exists in the file
                # the name is concatenated to previous parameter
                # example:
                #   CStarAt0Sbeox0Mm/Kg to CStarAt0 Sbeox0Mm/Kg (really two different params)
                line = re.sub(r'(\S)Sbeox', '\\1 Sbeox', line)

                names = line.split()
                skiprows = k + 2
                break
    if ftype == 'btl':
        # Capture stat names column.
        names.append('Statistic')
    metadata.update(
        {
            'header': header,
            'config': config,
            'names': names,
            'skiprows': skiprows,
            'time': time,
            'lon': lon,
            'lat': lat,
        }
    )
    return metadata


def asof(self, label):
    """pandas index workaround."""
    if label not in self:
        loc = self.searchsorted(label, side='left')
        if loc > 0:
            return self[loc - 1]
        else:
            return np.nan
    return label


class CTD(pd.DataFrame):
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
            pd.DataFrame(self),  # NOTE Using that type(data)==DataFrame and the
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

def calculate_sinking_velocity(row,cast):
    index = cast.index.get_loc(row.name)
 
    if index == 0:
        return 0.0
    prev_row = cast.iloc[index - 1]
    return abs(row['DepthCopy'] - prev_row['DepthCopy'])/abs(row['JD']-prev_row['JD'])
 
def calculate_julianday(mydate,mytime):
     
    # Calculate the julian date from the current datetime format (Norwegian time and date format)
    date_string = "%s %s"%(mydate,mytime)
    format = '%d.%m.%Y %H.%M.%S'
    my_date = datetime.strptime(date_string, format)
    return date2num(my_date,units="days since 1948-01-01 00:00:00",calendar="standard")
 
def from_saiv(fname, compression=None, below_water=False, lon=None,
             lat=None):
    """
    DataFrame constructor to open SAIV ASCII format. This file is modified by adding the
    header containing longitude and latitude. E.g.
    
    STATION;VT52;Kvinnherad;60.0096;5.954
    Ser;Meas;Sal.;Temp;T (FTU);Opt;Density;Depth(u);Date;Time;;   

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_saiv("/Users/trondkr/Dropbox/NIVA/OKOKYST_2017/OKOKYST_NS_Nord_Leon/t01_saiv_leon/SAIV_208_SN1380_VT79.txt")
    >>> fig, ax = cast['Temp'].plot()
    >>> _ = ax.axis([20, 24, 19, 0])
    >>> ax.grid(True)
    """
    separator=";"
    f = read_file(fname)
    header, names = [], []
    counter=0
    for k, line in enumerate(f.readlines()):
        line = line.strip()
       
        if counter==0:
           
            lat = line.split(separator)[3].strip()
            lon = line.split(separator)[4].strip()
            station=line.split(separator)[1]
            serial=line.split(separator)[0]
    

        if counter==1:
            header.append(line)
            col = line.split(separator)
            names=col
            

        if counter>1:
            skiprows = counter
            break
        counter+=1
 
    f.seek(0)
    skiprows=2
    cast = pd.read_table(f, header=None, index_col=None, names=names,
        skiprows=skiprows, sep=separator)
    f.close()
  

    cast['DepthCopy']=cast['Depth']
    cast.set_index('Depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    
    name = basename(fname)[1]
    if below_water:
        cast = remove_above_water(cast)

    pd.to_datetime(cast['Date'])
    
    cast['JD']=cast.apply(lambda row: calculate_julianday(row['Date'],row['Time']), axis=1)
    cast['dz/dtM']=abs(cast['DepthCopy'].astype('float64') - cast.shift()['DepthCopy'].astype('float64'))/(abs(cast['JD'].astype('float64')-cast.shift()['JD'].astype('float64'))*3600*24.)   #cast.apply(lambda row: calculate_sinking_velocity(row,cast), axis=1)
  
    return CTD(cast, longitude=float(lon), latitude=float(lat),serial=serial, name=station, header=header)

def from_edf(fname, below_water=False, lon=None, lat=None):
    """
    DataFrame constructor to open XBT EDF ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_edf(data_path.joinpath('XBT.EDF.gz'))
    >>> fig, ax = cast['temperature'].plot()
    >>> _ = ax.axis([20, 24, 19, 0])
    >>> ax.grid(True)

    """
    f = read_file(fname)
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
    cast = pd.read_table(
        f,
        header=None,
        index_col=None,
        names=names,
        skiprows=skiprows,
        delim_whitespace=True
    )
    f.close()

    cast.set_index('depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    name = basename(fname)[1]
    if below_water:
        cast = remove_above_water(cast)
    return CTD(cast, longitude=float(lon), latitude=float(lat),
               serial=serial, name=name, header=header)


def from_cnv(fname, below_water=False, time=None, lon=None, lat=None):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_cnv(data_path.joinpath('CTD_big.cnv.bz2'))
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['t090C'].plot()
    >>> ax.grid(True)

    """

    f = read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype='cnv')

    f.seek(0)
    cast = pd.read_fwf(
        f,
        header=None,
        index_col=None,
        names=metadata['names'],
        skiprows=metadata['skiprows'],
        delim_whitespace=True,
        widths=[11, ] * len(metadata['names'])
    )
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
        'flag': bool,
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
        time=metadata['time'],
        longitude=metadata['lon'],
        latitude=metadata['lat'],
        name=name,
        header=metadata['header'],
        config=metadata['config'],
        )


def from_bl(filename):
    """Read Seabird bottle-trip (bl) file

    df.time_of_reset provides the time when seasave was reset.

    Example
    -------
    >>> from ctd import DataFrame
    >>> df = DataFrame.from_bl(str(data_path.joinpath('bl', 'bottletest.bl')))
    >>> df.time_of_reset
    datetime.datetime(2018, 6, 25, 20, 8, 55)

    """
    df = pd.read_csv(filename, skiprows=2, parse_dates=[1], index_col=0,
                     names=["bottle_number", "time", "startscan", "endscan"])
    df.time_of_reset = pd.to_datetime(
        linecache.getline(filename, 2)[6:-1]).to_pydatetime()
    return df


def from_btl(fname, lon=None, lat=None):
    """
    DataFrame constructor to open Seabird CTD BTL-ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> bottles = DataFrame.from_btl(data_path.joinpath('btl', 'bottletest.btl'))

    """

    f = read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype='btl')

    f.seek(0)

    cast = pd.read_fwf(
        f,
        header=None,
        index_col=False,
        names=metadata['names'],
        parse_dates=False,
        skiprows=metadata['skiprows'],
    )
    f.close()

    # At this point the data frame is not correctly lined up (multiple rows
    # for avg, std, min, max or just avg, std, etc).
    # Also needs date,time,and bottle number to be converted to one per line.

    # Get row types, see what you have: avg, std, min, max or just avg, std.
    rowtypes = cast[cast.columns[-1]].unique()
    # Get times and dates which occur on second line of each bottle.
    dates = cast.iloc[::len(rowtypes), 1].reset_index(drop=True)
    times = cast.iloc[1::len(rowtypes), 1].reset_index(drop=True)
    datetimes = dates + ' ' + times

    # Fill the Date column with datetimes.
    cast.loc[::len(rowtypes), 'Date'] = datetimes.values
    cast.loc[1::len(rowtypes), 'Date'] = datetimes.values

    # Fill missing rows.
    cast['Bottle'] = cast['Bottle'].fillna(method='ffill')
    cast['Date'] = cast['Date'].fillna(method='ffill')

    cast['Statistic'] = cast['Statistic'].str.replace(r'\(|\)', '')  # (avg) to avg

    name = basename(fname)[0]

    dtypes = {
        'bpos': int,
        'pumps': bool,
        'flag': bool,
        'Bottle': int,
        'Scan': int,
        'Statistic': str,
        'Date': str,
    }
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = cast[column].astype(float)
            except ValueError:
                warnings.warn('Could not convert %s to float.' % column)

    cast['Date'] = pd.to_datetime(cast['Date'])

    return CTD(
        cast,
        longitude=metadata['lon'],
        latitude=metadata['lat'],
        name=name,
        header=metadata['header'],
        config=metadata['config']
    )


def from_fsi(fname, skiprows=9, below_water=False, lon=None, lat=None):
    """
    DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.

    Examples
    --------
    >>> from ctd import DataFrame
    >>> cast = DataFrame.from_fsi(data_path.joinpath('FSI.txt.gz'))
    >>> downcast, upcast = cast.split()
    >>> fig, ax = downcast['TEMP'].plot()
    >>> ax.grid(True)

    """
    f = read_file(fname)
    cast = pd.read_table(
        f,
        header='infer',
        index_col=None,
        skiprows=skiprows,
        dtype=float,
        delim_whitespace=True
    )
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
    >>> fname = data_path.joinpath('CTD/g01l01s01.ros')
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
