# -*- coding: utf-8 -*-
#
# utilities.py
#
# purpose:  Utilities functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  23-Jul-2013
# modified: Thu 25 Jul 2013 01:13:03 PM BRT
#
# obs:
#

# Standard library.
import os
import bz2
import gzip
import zipfile
from cStringIO import StringIO
from xml.etree import cElementTree as etree

# Scientific stack.
import numpy as np


# Utilities.
def header(xml):
    return etree(xml)


def basename(fname):
    """Return filename without path.

    Examples
    --------
    >>> fname = '../test/data/FSI.txt.zip'
    >>> basename(fname)
    ('../test/data', 'FSI.txt', '.zip')
    """
    path, name = os.path.split(fname)
    name, ext = os.path.splitext(name)
    return path, name, ext


def rolling_window(data, block):
    """
    http://stackoverflow.com/questions/4936620/
    Using strides for an efficient moving average filter.
    """
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def extrap1d(interpolator):
    """
    http://stackoverflow.com/questions/2745329/
    How to make scipy.interpolate give an extrapolated result beyond the
    input range.
    """
    xs, ys = interpolator.x, interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return (ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) /
                    (xs[-1] - xs[-2]))
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def normalize_names(name):
    name = name.strip()
    name = name.strip('*')
    return name


def read_file(fname, compression=None):
    if compression == 'gzip':
        lines = gzip.open(fname)
    elif compression == 'bz2':
        lines = bz2.BZ2File(fname)
    elif compression == 'zip':
        zfile = zipfile.ZipFile(fname)
        # Zip format might contain more than one file in the archive (similar
        # to tar), here we assume that there is just one file per zipfile.
        name = zfile.namelist()[0]
        lines = StringIO(zfile.read(name))
    else:
        lines = open(fname)
    return lines
