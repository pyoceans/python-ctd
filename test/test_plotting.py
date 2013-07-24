# -*- coding: utf-8 -*-
#
# test_processing.py
#
# purpose:  Test processing step from ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Fri 19 Jul 2013 01:48:30 PM BRT
#
# obs:
#


import unittest
import cStringIO

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from numpy.testing import assert_array_equal

from ctd import DataFrame


def compare_images(fig, figname):
    imgdata = cStringIO.StringIO()
    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.savefig(imgdata, dpi=75, format='png')
    imgdata.seek(0)
    im1 = plt.imread(imgdata)
    im2 = plt.imread(figname)
    assert_array_equal(im1, im2, err_msg='Image arrays differ.')


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

    def test_xbt_plot(self):
        fig, ax = self.xbt['temperature'].plot()
        compare_images(fig, figname='data/test_plot_xbt.png')

    def test_cnv_temperature(self):
        fig, ax = self.cnv['t090c'].plot()
        compare_images(fig, figname='data/test_plot_cnv.png')

    def test_fsi_plot_vars(self):
        fig, ax = self.fsi.plot_vars(['TEMP', 'SAL*'])
        compare_images(fig, figname='data/test_plot_vars_fsi.png')


class AdvancedPlotting(unittest.TestCase):
    """
    fig, ax, cb = plot_section(Temp, inverse=True, filled=False)
    fig, ax, cb = plot_section(Temp, inverse=True, filled=True)
    fig, ax, cb = plot_section(Sal, inverse=True)
    fig, ax, cb = plot_section(Sal, inverse=False)
    """
    pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
