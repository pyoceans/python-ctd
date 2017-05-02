from __future__ import (absolute_import, division, print_function)

import os
import re
import unittest

from collections import OrderedDict
from glob import glob

from ctd import DataFrame, Series

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from pandas import Panel

matplotlib.use('Agg', warn=True)

data_path = os.path.join(os.path.dirname(__file__), 'data')


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


def alphanum_key(s):
    key = re.split(r'(\d+)', s)
    key[1::2] = list(map(int, key[1::2]))
    return key


def proc_ctd(fname):
    """
    CTD processing.

    """
    cast = DataFrame.from_cnv(fname, compression='gzip',
                              below_water=True).split()[0]
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]
    name = os.path.basename(fname).split('.')[0]
    # Removed unwanted columns.
    keep = {'t090C', 't190C', 'longitude', 'latitude'}
    drop = keep.symmetric_difference(cast.columns)
    cast.drop(drop, axis=1, inplace=True)
    cast = cast.apply(Series.bindata, **dict(delta=1.))
    cast = cast.apply(Series.interpolate)
    cast.name = name
    return cast


def make_section(data_path=data_path, variable='t090C'):
    lon, lat = [], []
    section = OrderedDict()
    pattern = '%s/CTD/g01mcan*c.cnv.gz' % data_path
    fnames = sorted(glob(pattern), key=alphanum_key)
    for fname in fnames:
        cast = proc_ctd(fname)
        name = os.path.basename(fname).split('.')[0]
        section.update({name: cast})
        lon.append(cast.longitude.mean())
        lat.append(cast.latitude.mean())

    section = Panel.fromDict(section)
    section = section.minor_xs(variable)
    section.lon, section.lat = lon, lat
    return section


class BasicPlotting(unittest.TestCase):
    def setUp(self):
        self.xbt = DataFrame.from_edf('{}/{}'.format(data_path, 'XBT.EDF.zip'),
                                      compression='zip')
        self.fsi = DataFrame.from_fsi('{}/{}'.format(data_path, 'FSI.txt.gz'),
                                      compression='gzip', skiprows=9)
        self.cnv = DataFrame.from_cnv('{}/{}'.format(data_path,
                                                     'small.cnv.bz2'),
                                      compression='bz2')

    def tearDown(self):
        plt.close('all')

    def test_xbt_plot(self):
        _check_plot_works(self.xbt['temperature'].plot)

    def test_cnv_temperature(self):
        _check_plot_works(self.cnv['t090C'].plot)

    def test_fsi_plot_vars(self):
        _check_plot_works(self.fsi.plot_vars, variables=['TEMP', 'SAL*'])


class AdvancedPlotting(unittest.TestCase):
    def setUp(self):
        self.t090C = make_section(data_path=data_path, variable='t090C')

    def tearDown(self):
        plt.close('all')

    def test_section_reverse(self):
        _check_section_works(self.t090C.plot_section, reverse=True)

    def test_section_reverse_filled(self):
        _check_section_works(self.t090C.plot_section, reverse=True,
                             filled=True)

    def test_section(self):
        _check_section_works(self.t090C.plot_section)

    def test_section_filled(self):
        _check_section_works(self.t090C.plot_section, filled=True)
