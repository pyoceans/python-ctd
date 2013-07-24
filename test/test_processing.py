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

from ctd import DataFrame


class DataFrameTests(unittest.TestCase):
    def setUp(self):
        self.xbt = DataFrame.from_edf(fname='data/XBT.EDF.zip',
                                      compression='zip')
        self.fsi = DataFrame.from_fsi(fname='data/FSI.txt.gz',
                                      compression='gzip', skiprows=9)
        self.cnv = DataFrame.from_cnv(fname='data/CTD_big.cnv.bz2',
                                      compression='bz2')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_split_cnv(self):
        self.assertIsInstance(self.cnv.split(), tuple)

    def test_split_fsi(self):
        self.assertIsInstance(self.fsi.split(), tuple)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
