from __future__ import absolute_import, unicode_literals

import re
import os
import nose
import unittest
from io import StringIO

from glob import glob
try:
    from collections import OrderedDict
except ImportError:
    pass

from pandas import Panel
from ctd.utilities import read_file
from ctd import (DataFrame, Series, rosette_summary, lp_filter, movingaverage,
                 derive_cnv, plot_section)

data_path = os.path.join(os.path.dirname(__file__), 'data')


def alphanum_key(s):
    key = re.split(r"(\d+)", s)
    key[1::2] = list(map(int, key[1::2]))
    return key


def proc_ctd(fname, compression='gzip', below_water=True):
    """
    Quick `proc_ctd` function.

    """
    # 00-Split, clean 'bad pump' data, and apply flag.

    cast = DataFrame.from_cnv(fname, compression=compression,
                              below_water=below_water).split()[0]

    name = os.path.basename(fname).split('.')[0]
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]  # True for bad values.

    # Smooth velocity with a 2 seconds windows.
    cast['dz/dtM'] = movingaverage(cast['dz/dtM'], window_size=48)

    # 01-Filter pressure.
    kw = dict(sample_rate=24.0, time_constant=0.15)
    cast.index = lp_filter(cast.index, **kw)

    # 02-Remove pressure reversals.
    cast = cast.press_check()
    cast = cast.dropna()

    # 03-Loop Edit.
    cast = cast[cast['dz/dtM'] >= 0.25]  # Threshold velocity.

    # 04-Remove spikes.
    kw = dict(n1=2, n2=20, block=100)
    cast = cast.apply(Series.despike, **kw)

    # 05-Bin-average.
    cast = cast.apply(Series.bindata, **dict(delta=1.))

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
        kw = dict(window_len=window_len, window='hanning')
        cast = cast.apply(Series.smooth, **kw)

    # 08-Derive.
    cast.lat = cast['latitude'].mean()
    cast.lon = cast['longitude'].mean()
    cast = derive_cnv(cast)
    cast.name = name
    return cast


class ReadFile(unittest.TestCase):
    def test_zip(self):
        cfile = read_file('{}/{}'.format(data_path, 'XBT.EDF.zip'),
                          compression='zip')
        self.assertIsInstance(cfile, StringIO)

    def test_gzip(self):
        cfile = read_file('{}/{}'.format(data_path, 'XBT.EDF.gz'),
                          compression='gzip')
        self.assertIsInstance(cfile, StringIO)

    def test_bz2(self):
        cfile = read_file('{}/{}'.format(data_path, 'XBT.EDF.bz2'),
                          compression='bz2')
        self.assertIsInstance(cfile, StringIO)

    def test_uncompresed(self):
        cfile = read_file('{}/{}'.format(data_path, 'XBT.EDF'),
                          compression=None)
        self.assertIsInstance(cfile, StringIO)


class DataFrameTests(unittest.TestCase):
    def setUp(self):
        self.xbt = DataFrame.from_edf('{}/{}'.format(data_path,
                                                     'XBT.EDF.zip'),
                                      compression='zip')
        self.fsi = DataFrame.from_fsi('{}/{}'.format(data_path,
                                                     'FSI.txt.gz'),
                                      compression='gzip', skiprows=9)
        self.cnv = DataFrame.from_cnv('{}/{}'.format(data_path,
                                                     'CTD_big.cnv.bz2'),
                                      compression='bz2')
        self.ros = rosette_summary('{}/{}'.format(data_path,
                                                  'CTD/g01l03s01m-m2.ros'))

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    """Check if a DataFrame is returned."""
    def test_fsi_is_dataframe(self):
        self.assertIsInstance(self.fsi, DataFrame)

    def test_xbt_is_dataframe(self):
        self.assertIsInstance(self.xbt, DataFrame)

    def test_cnv_is_dataframe(self):
        self.assertIsInstance(self.cnv, DataFrame)

    """Check if DataFrame is not empty."""
    def test_fsi_is_not_empty(self):
        self.assertFalse(self.fsi.empty)

    def test_xbt_is_not_empty(self):
        self.assertFalse(self.xbt.empty)

    def test_cnv_is_not_empty(self):
        self.assertFalse(self.cnv.empty)


class HeaderTest(unittest.TestCase):
    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_header_parse(self):
        lon, lat = '-42', '42'
        xbt1 = DataFrame.from_edf('{}/{}'.format(data_path, 'C3_00005.edf'),
                                  lon=lon, lat=lat)
        self.assertTrue((xbt1.longitude, xbt1.latitude) ==
                        (float(lon), float(lat)))

        with self.assertRaises(ValueError):
            DataFrame.from_edf('{}/{}'.format(data_path,
                                              'C3_00005.edf'),
                               lon=None, lat=None)

    def test_pressure_field_labels(self):
        """
        Support different pressure field labels encountered in
        Sea-Bird cnv files (issue #3)

        """
        for fname in sorted(glob('{}/{}'.format(data_path,
                                                'issue3prlabworks*.cnv'))):
            DataFrame.from_cnv(fname)
        for fname in sorted(glob('{}/{}'.format(data_path,
                                                'issue3prlabfails*.cnv'))):
            with self.assertRaises(KeyError):
                DataFrame.from_cnv(fname)


class SectionTest(unittest.TestCase):
    def setUp(self):
        lon, lat = [], []
        pattern = '{}/{}'.format(data_path, 'CTD/g01mcan*c.cnv.gz')
        fnames = sorted(glob(pattern), key=alphanum_key)
        try:
            section = OrderedDict()
        except:
            raise nose.SkipTest
        for fname in fnames:
            cast = proc_ctd(fname)
            name = os.path.basename(fname).split('.')[0]
            section.update({name: cast})
            lon.append(cast.longitude.mean())
            lat.append(cast.latitude.mean())

        # Section.
        self.section = Panel.fromDict(section)
        self.lon, self.lat = lon, lat

    def test_section(self):
        CT = self.section.minor_xs('CT')
        CT.lon, CT.lat = self.lon, self.lat
        fig, ax, cb = plot_section(CT, reverse=True)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
