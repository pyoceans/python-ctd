from __future__ import (absolute_import, division, print_function)

import os
import re
import unittest

from collections import OrderedDict
from glob import glob

from ctd import DataFrame, Series, derive_cnv, lp_filter, movingaverage

import numpy as np

data_path = os.path.join(os.path.dirname(__file__), 'data')


def alphanum_key(s):
    key = re.split(r'(\d+)', s)
    key[1::2] = list(map(int, key[1::2]))
    return key


def proc_ctd(fname, compression='gzip', below_water=True):
    """
    CTD processing steps.

    """
    # 00-Split, clean 'bad pump' data, and apply flag.
    cast = DataFrame.from_cnv(fname, compression=compression,
                              below_water=below_water).split()[0]
    if 'pumps' in cast.columns:
        # True for good values.
        cast = cast[cast['pumps']]
    if 'flag' in cast.columns:
        # True for bad values.
        cast = cast[~cast['flag']]
    name = os.path.basename(fname).split('.')[0]

    # Removed unwanted columns.
    keep = {'altM', 'c0S/m', 'dz/dtM', 'wetCDOM', 'latitude', 'longitude',
            'sbeox0Mm/Kg', 'sbeox1Mm/Kg', 'oxsolMm/Kg', 'oxsatMm/Kg', 'par',
            'pla', 'sva', 't090C', 't190C', 'tsa', 'sbeox0V'}

    null = list(map(cast.pop, keep.symmetric_difference(cast.columns)))
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


class BasicProcessingTests(unittest.TestCase):
    def setUp(self):
        name = 'CTD-spiked-unfiltered.cnv.bz2'
        self.raw = DataFrame.from_cnv('{}/{}'.format(data_path, name),
                                      compression='bz2')
        name = 'CTD-spiked-filtered.cnv.bz2'
        self.prc = DataFrame.from_cnv('{}/{}'.format(data_path, name),
                                      compression='bz2')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # Split.
    def test_split_return_tuple(self):
        self.assertIsInstance(self.raw.split(), tuple)

    def test_split_cnv(self):
        downcast, upcast = self.raw.split()
        self.assertTrue(downcast.index.size + upcast.index.size ==
                        self.raw.index.size)

    # Despike.
    def test_despike(self):
        dirty = self.prc['c0S/m'].split()[0]  # Looking at downcast only.
        clean = dirty.despike(n1=2, n2=20, block=500)
        spikes = clean.isnull()
        equal = (dirty[~spikes] == clean[~spikes]).all()
        self.assertTrue(spikes.any() and equal)

    # Filter.
    def test_lp_filter(self):
        kw = dict(sample_rate=24.0, time_constant=0.15)
        unfiltered = self.raw.index.values
        filtered = lp_filter(unfiltered, **kw)
        # FIXME: Not a good test...
        np.testing.assert_almost_equal(filtered, self.prc.index.values,
                                       decimal=1)

    # Pressure check.
    def test_press_check(self):
        unchecked = self.raw['t090C']
        press_checked = unchecked.press_check()
        reversals = press_checked.isnull()
        equal = (unchecked[~reversals] == press_checked[~reversals]).all()
        self.assertTrue(reversals.any() and equal)

    def test_bindata(self):
        delta = 1.
        down = self.prc['t090C'].split()[0]
        down = down.bindata(delta=delta)
        self.assertTrue(np.unique(np.diff(down.index.values)) == delta)

    # PostProcessingTests.
    def test_smooth(self):
        pass  # TODO

    def test_mixed_layer_depth(self):
        pass  # TODO

    def test_barrier_layer_thickness(self):
        pass  # TODO

    def derive_cnv(self):
        derived = derive_cnv(self.raw)
        new_cols = set(derived).symmetric_difference(self.raw.columns)
        self.assertTrue(['CT', 'SA', 'SP', 'SR', 'sigma0_CT', 'z'] ==
                        sorted(new_cols))


class AdvancedProcessingTests(unittest.TestCase):
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
    # TODO: Write section tests.


if __name__ == '__main__':
    unittest.main()
