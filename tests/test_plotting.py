from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ctd import DataFrame, Series
from ctd.utilities import Path

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from pandas import Panel

import pytest


matplotlib.use('Agg', warn=True)

data_path = Path(__file__).parent.joinpath('data')


def assert_is_valid_plot_return_object(objs):
    if isinstance(objs, np.ndarray):
        for el in objs.flat:
            assert isinstance(el, plt.Axes), ('one of \'objs\' is not a '
                                              'matplotlib Axes instance, '
                                              'type encountered {0!r}'
                                              ''.format(el.__class__.__name__))
    else:
        assert isinstance(objs, (plt.Artist, tuple, dict)), (
            'objs is neither an ndarray of Artist instances nor a '
            'single Artist instance, tuple, or dict, "objs" is a {0!r} '
            ''.format(objs.__class__.__name__))


def _check_plot_works(f, *args, **kwargs):
    fig, ax = f(*args, **kwargs)

    assert fig is not None
    assert_is_valid_plot_return_object(ax)
    plt.close()


def _check_section_works(f, **kwargs):
    fig, ax, cb = f(**kwargs)

    assert fig is not None
    assert_is_valid_plot_return_object(ax)
    plt.close()


def proc_ctd(fname):
    """
    CTD processing.

    """
    cast = DataFrame.from_cnv(fname, below_water=True).split()[0]
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]
    name = Path(fname).stem
    # Removed unwanted columns.
    keep = {'t090C', 't190C', 'longitude', 'latitude'}
    drop = keep.symmetric_difference(cast.columns)
    cast.drop(drop, axis=1, inplace=True)
    cast = cast.apply(Series.bindata, **{'delta': 1.})
    cast = cast.apply(Series.interpolate)
    cast.name = name
    return cast


def make_section(data_path=data_path, variable='t090C'):
    lon, lat = [], []
    section = OrderedDict()
    fnames = sorted(data_path.joinpath('CTD').glob('g01mcan*c.cnv.gz'))
    for fname in fnames:
        cast = proc_ctd(fname)
        name = fname.stem
        section.update({name: cast})
        lon.append(cast['longitude'].mean())
        lat.append(cast['latitude'].mean())

    section = Panel.fromDict(section)
    section = section.minor_xs(variable)
    section.lon, section.lat = lon, lat
    return section


# BasicPlotting
@pytest.fixture
def xbt():
    yield DataFrame.from_edf(data_path.joinpath('XBT.EDF.zip'))
    plt.close('all')


@pytest.fixture
def fsi():
    yield DataFrame.from_fsi(data_path.joinpath('FSI.txt.gz'), skiprows=9)
    plt.close('all')


@pytest.fixture
def cnv():
    yield DataFrame.from_cnv(data_path.joinpath('small.cnv.bz2'))
    plt.close('all')


def test_xbt_plot(xbt):
    _check_plot_works(xbt['temperature'].plot)


def test_cnv_temperature(cnv):
    _check_plot_works(cnv['t090C'].plot)


def test_fsi_plot_vars(fsi):
    _check_plot_works(fsi.plot_vars, variables=['TEMP', 'SAL*'])


# AdvancedPlotting
@pytest.fixture
def t090c():
    yield make_section(data_path=data_path, variable='t090C')
    plt.close('all')


def test_section_reverse(t090c):
    _check_section_works(t090c.plot_section, reverse=True)


def test_section_reverse_filled(t090c):
    _check_section_works(t090c.plot_section, reverse=True, filled=True)


def test_section(t090c):
    _check_section_works(t090c.plot_section)


def test_section_filled(t090c):
    _check_section_works(t090c.plot_section, filled=True)
