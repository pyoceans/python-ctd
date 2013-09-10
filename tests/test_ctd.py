# -*- coding: utf-8 -*-
#
# test_ctd.py
#
# purpose:  Test basic read/load and save from ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Thu 22 Aug 2013 01:20:51 PM BRT
#
# obs: TODO: to_nc test.
#

import re
import os
import bz2
import gzip
import nose
import unittest
import cStringIO

from glob import glob
try:
    from collections import OrderedDict
except ImportError:
    pass

from pandas import Panel
from ctd.utilities import read_file
from ctd import (DataFrame, Series, rosette_summary, lp_filter, movingaverage,
                 derive_cnv, plot_section)


def alphanum_key(s):
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key


def proc_ctd(fname, compression='gzip', below_water=True):
    """Quick `proc_ctd` function."""
    # 00-Split, clean 'bad pump' data, and apply flag.
    cast = DataFrame.from_cnv(fname, compression=compression,
                              below_water=below_water).split()[0]
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]  # True for bad values.
    name = os.path.basename(fname).split('.')[0]

    # Removed unwanted columns.
    keep = set(['altM', 'c0S/m', 'dz/dtM', 'wetCDOM', 'latitude',
                'longitude', 'sbeox0Mm/Kg', 'sbeox1Mm/Kg', 'oxsolMm/Kg',
                'oxsatMm/Kg', 'par', 'pla', 'sva', 't090C', 't190C', 'tsa',
                'sbeox0V'])

    null = map(cast.pop, keep.symmetric_difference(cast.columns))
    del null

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


class ReadCompressedFile(unittest.TestCase):
    """Pandas can read StringI, GzipFile, BZ2File and file types."""

    def test_zip(self):
        """cStringIO.StringI type."""
        cfile = read_file('data/XBT.EDF.zip', compression='zip')
        self.assertIsInstance(cfile, cStringIO.InputType)

    def test_gzip(self):
        """GzipFile type."""
        cfile = read_file('data/XBT.EDF.gz', compression='gzip')
        self.assertIsInstance(cfile, gzip.GzipFile)

    def test_bz2(self):
        """bz2.BZ2File type."""
        cfile = read_file('data/XBT.EDF.bz2', compression='bz2')
        self.assertIsInstance(cfile, bz2.BZ2File)

    def test_uncompresed(self):
        """file type."""
        cfile = read_file('data/XBT.EDF', compression=None)
        self.assertIsInstance(cfile, file)


class DataFrameTests(unittest.TestCase):
    def setUp(self):
        self.xbt = DataFrame.from_edf(fname='data/XBT.EDF.zip',
                                      compression='zip')
        self.fsi = DataFrame.from_fsi(fname='data/FSI.txt.gz',
                                      compression='gzip', skiprows=9)
        self.cnv = DataFrame.from_cnv(fname='data/CTD_big.cnv.bz2',
                                      compression='bz2')
        self.ros = rosette_summary(fname='data/CTD/g01l03s01m-m2.ros')

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


class SectionTest(unittest.TestCase):
    def setUp(self):
        lon, lat = [], []
        pattern = './data/CTD/g01mcan*c.cnv.gz'
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
        self.section.lon, self.section.lat = lon, lat

    def test_section(self):
        CT = self.section.minor_xs('CT')
        CT.lon, CT.lat = self.lon, self.lat
        fig, ax, cb = plot_section(CT, reverse=True)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
