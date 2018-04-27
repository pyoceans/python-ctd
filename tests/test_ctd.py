from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from io import StringIO

from ctd import (
    DataFrame,
    Series,
    derive_cnv,
    lp_filter,
    movingaverage,
    plot_section,
    rosette_summary
    )
from ctd.utilities import Path, read_file

from pandas import Panel

import pytest

data_path = Path(__file__).parent.joinpath('data')


def proc_ctd(fname, below_water=True):
    """
    Quick `proc_ctd` function.

    """
    # 00-Split, clean 'bad pump' data, and apply flag.

    cast = DataFrame.from_cnv(
        fname,
        below_water=below_water
        ).split()[0]

    name = Path(fname).stem
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]  # True for bad values.

    # Smooth velocity with a 2 seconds windows.
    cast['dz/dtM'] = movingaverage(cast['dz/dtM'], window_size=48)

    # 01-Filter pressure.
    kw = {
        'sample_rate': 24.0,
        'time_constant': 0.15
    }
    cast.index = lp_filter(cast.index, **kw)

    # 02-Remove pressure reversals.
    cast = cast.press_check()
    cast = cast.dropna()

    # 03-Loop Edit.
    cast = cast[cast['dz/dtM'] >= 0.25]  # Threshold velocity.

    # 04-Remove spikes.
    kw = {
        'n1': 2,
        'n2': 20,
        'block': 100
    }
    cast = cast.apply(Series.despike, **kw)

    # 05-Bin-average.
    cast = cast.apply(Series.bindata, **{'delta': 1.})

    # 06-interpolate.
    cast = cast.apply(Series.interpolate)

    if False:
        # 07-Smooth.
        pmax = max(cast.index)
        if pmax >= 500.:
            window_len = 21
        elif pmax >= 100.:
            window_len = 11
        else:
            window_len = 5
            kw = {
                'window_len': window_len,
                'window': 'hanning'
            }
        cast = cast.apply(Series.smooth, **kw)

    # 08-Derive.
    cast.lat = cast['latitude'].mean()
    cast.lon = cast['longitude'].mean()
    cast = derive_cnv(cast)
    cast.name = name
    return cast


# Test read file
def test_zip():
    cfile = read_file(data_path.joinpath('XBT.EDF.zip'))
    assert isinstance(cfile, StringIO)


def test_gzip():
    cfile = read_file(data_path.joinpath('XBT.EDF.gz'))
    assert isinstance(cfile, StringIO)


def test_bz2():
    cfile = read_file(data_path.joinpath('XBT.EDF.bz2'))
    assert isinstance(cfile, StringIO)


def test_uncompresed():
    cfile = read_file(data_path.joinpath('XBT.EDF'))
    assert isinstance(cfile, StringIO)


# DataFrame
@pytest.fixture
def xbt():
    yield DataFrame.from_edf(data_path.joinpath('XBT.EDF.zip'))


@pytest.fixture
def fsi():
    yield DataFrame.from_fsi(data_path.joinpath('FSI.txt.gz'), skiprows=9)


@pytest.fixture
def cnv():
    yield DataFrame.from_cnv(data_path.joinpath('small.cnv.bz2'))


@pytest.fixture
def btl():
    yield DataFrame.from_btl(data_path.joinpath('btl', 'bottletest.btl'))


@pytest.fixture
def ros():
    yield rosette_summary(data_path.joinpath('CTD', 'g01l03s01m-m2.ros'))


# Check if a DataFrame is returned.
def test_xbt_is_dataframe(xbt):
    assert isinstance(xbt, DataFrame)


def test_fsi_is_dataframe(fsi):
    assert isinstance(fsi, DataFrame)


def test_cnv_is_dataframe(cnv):
    assert isinstance(cnv, DataFrame)


def test_btl_is_dataframe(btl):
    assert isinstance(btl, DataFrame)


def test_ros_is_dataframe(ros):
    assert isinstance(ros, DataFrame)


# Check if DataFrame is not empty
def test_xbt_is_not_empty(xbt):
    assert not xbt.empty


def test_fsi_is_not_empty(fsi):
    assert not fsi.empty


def test_cnv_is_not_empty(cnv):
    assert not cnv.empty


def test_btl_is_not_empty(btl):
    assert not btl.empty


def test_ros_is_not_empty(ros):
    assert not ros.empty


# HeaderTest
def test_header_parse():
    lon, lat = '-42', '42'
    xbt1 = DataFrame.from_edf(data_path.joinpath('C3_00005.edf'), lon=lon, lat=lat)
    assert (xbt1.longitude, xbt1.latitude) == (float(lon), float(lat))

    with pytest.raises(ValueError):
        DataFrame.from_edf(data_path.joinpath('C3_00005.edf'), lon=None, lat=None)


def test_pressure_field_labels():
    """
    Support different pressure field labels encountered in
    Sea-Bird cnv files (issue #3)

    """
    for fname in sorted(data_path.glob('issue3prlabworks*.cnv')):
        DataFrame.from_cnv(fname)
    for fname in sorted(data_path.glob('issue3prlabfails*.cnv')):
        with pytest.raises(KeyError):
            DataFrame.from_cnv(fname)


@pytest.fixture
def section():
    lon, lat = [], []
    fnames = sorted(data_path.glob('CTD/g01mcan*c.cnv.gz'))
    section = OrderedDict()
    for fname in fnames:
        cast = proc_ctd(fname)
        name = Path(fname).stem
        section.update({name: cast})
        lon.append(cast.longitude.mean())
        lat.append(cast.latitude.mean())

    # Section.
    section = Panel.fromDict(section)
    return {'section': section, 'lon': lon, 'lat': lat}


def test_section():
    data = section()
    CT = data['section'].minor_xs('CT')
    CT.lon, CT.lat = data['lon'], data['lat']
    fig, ax, cb = plot_section(CT, reverse=True)
