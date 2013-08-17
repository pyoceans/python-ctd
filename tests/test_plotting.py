# -*- coding: utf-8 -*-
#
# test_processing.py
#
# purpose:  Test processing step from ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Sat 17 Aug 2013 04:49:58 PM BRT
#
# obs:
#

import os
import re
import unittest
from glob import glob
from collections import OrderedDict

from pandas import Panel
import matplotlib.pyplot as plt

from ctd import (DataFrame, Series, movingaverage, lp_filter, derive_cnv,
                 plot_section)

plt.switch_backend('Agg')
from matplotlib.testing.decorators import image_comparison

path = os.path.join(os.path.dirname(__file__), 'plot_results')
data_path = os.path.join(os.path.dirname(__file__), 'data')


def alphanum_key(s):
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key


def proc_ctd(fname, compression='gzip', below_water=True):
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


class PlotUtilities(unittest.TestCase):
    # TODO: get_maxdepth, extrap_sec, gen_topomask,
    pass


class BasicPlotting(unittest.TestCase):
    def setUp(self):
        self.xbt = DataFrame.from_edf(fname='data/XBT.EDF.zip',
                                      compression='zip')
        self.fsi = DataFrame.from_fsi(fname='data/FSI.txt.gz',
                                      compression='gzip', skiprows=9)
        self.cnv = DataFrame.from_cnv(fname='data/CTD_big.cnv.bz2',
                                      compression='bz2')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    @image_comparison(baseline_images=['%s/test_xbt_plot.png' % path])
    def test_xbt_plot(self):
        fig, ax = self.xbt['temperature'].plot()

    @image_comparison(baseline_images=['%s/test_cnv_temperature.png' % path])
    def test_cnv_temperature(self):
        fig, ax = self.cnv['t090C'].plot()

    @image_comparison(baseline_images=['%s/test_fsi_plot_vars.png' % path])
    def test_fsi_plot_vars(self):
        fig, ax = self.fsi.plot_vars(['TEMP', 'SAL*'])


class AdvancedPlotting(unittest.TestCase):
    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def setUp(self):
        lon, lat = [], []
        pattern = '%s/CTD/g01mcan*c.cnv.gz' % data_path
        fnames = sorted(glob(pattern), key=alphanum_key)
        section = OrderedDict()
        for fname in fnames:
            cast = proc_ctd(fname)
            name = os.path.basename(fname).split('.')[0]
            section.update({name: cast})
            lon.append(cast.longitude.mean())
            lat.append(cast.latitude.mean())

        # Section (FIXME: Propagate lon, lat with MetaDataFrame).
        self.section = Panel.fromDict(section)
        self.CT = self.section.minor_xs('CT')
        self.CT.lon, self.CT.lat = lon, lat

    @image_comparison(baseline_images=['%s/test_section_reverse.png' % path])
    def test_section_reverse(self):
        fig, ax, cb = plot_section(self.CT, reverse=True)

    @image_comparison(baseline_images=['%s/test_section_reverse_filled.png' %
                                       path])
    def test_section_reverse_filled(self):
        fig, ax, cb = plot_section(self.CT, reverse=True, filled=True)

    @image_comparison(baseline_images=['%s/test_section.png' % path])
    def test_section(self):
        fig, ax, cb = plot_section(self.CT)

    @image_comparison(baseline_images=['%s/test_section_filled.png' % path])
    def test_section_filled(self):
        fig, ax, cb = plot_section(self.CT, filled=True)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
