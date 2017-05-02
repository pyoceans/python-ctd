from __future__ import (absolute_import, division, print_function)

import bz2
import gzip
import os
import zipfile

from io import StringIO

from xml.etree import cElementTree as etree

import numpy as np


def header(xml):
    return etree(xml)


def basename(fname):
    """
    Return file name without path.

    Examples
    --------
    >>> fname = '../test/data/FSI.txt.zip'
    >>> print('{}, {}, {}'.format(*basename(fname)))
    ../test/data, FSI.txt, .zip

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
    How to make scipy.interpolate return an extrapolated result beyond the
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
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def normalize_names(name):
    name = name.strip()
    name = name.strip('*')
    return name


def read_file(fname, compression=None):
    if compression == 'gzip':
        cfile = gzip.open(fname)
    elif compression == 'bz2':
        cfile = bz2.BZ2File(fname)
    elif compression == 'zip':
        # NOTE: Zip format may contain more than one file in the archive
        # (similar to tar), here we assume that there is just one file per
        # zipfile!  Also, we ask for the name because it can be different from
        # the zipfile file!!
        zfile = zipfile.ZipFile(fname)
        name = zfile.namelist()[0]
        cfile = zfile.open(name)
    else:
        cfile = open(fname, 'rb')
    text = cfile.read().decode(encoding='utf-8', errors='replace')
    cfile.close()
    return StringIO(text)
