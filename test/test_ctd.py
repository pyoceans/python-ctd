# -*- coding: utf-8 -*-
#
# test_ctd.py
#
# purpose:  Test basic read/load and save from ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Sat 20 Jul 2013 05:02:14 PM BRT
#
# obs: TODO: to_nc test.
#

import bz2
import gzip
import unittest
import cStringIO

from ctd.ctd import read_file
from ctd import DataFrame, rosette_summary


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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
