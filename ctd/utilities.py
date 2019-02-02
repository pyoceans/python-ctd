import bz2
import gzip
import re
import zipfile
from io import StringIO
from xml.etree import cElementTree as etree

import numpy as np

try:
    from pathlib import Path
except ImportError as e:
    from pathlib2 import Path

path_type = (Path,)


def header(xml):
    return etree(xml)


def basename(fname):
    """Return file name without path."""
    if not isinstance(fname, path_type):
        fname = Path(fname)
    path, name, ext = fname.parent, fname.stem, fname.suffix
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
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (
                xs[-1] - xs[-2]
            )
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def normalize_names(name):
    name = name.strip()
    name = name.strip("*")
    return name


def _open_compressed(fname):
    extension = fname.suffix.lower()
    if extension in [".gzip", ".gz"]:
        cfile = gzip.open(str(fname))
    elif extension == ".bz2":
        cfile = bz2.BZ2File(str(fname))
    elif extension == ".zip":
        # NOTE: Zip format may contain more than one file in the archive
        # (similar to tar), here we assume that there is just one file per
        # zipfile!  Also, we ask for the name because it can be different from
        # the zipfile file!!
        zfile = zipfile.ZipFile(str(fname))
        name = zfile.namelist()[0]
        cfile = zfile.open(name)
    else:
        raise ValueError(
            "Unrecognized file extension. Expected .gzip, .bz2, or .zip, got {}".format(
                extension
            )
        )
    contents = cfile.read()
    cfile.close()
    return contents


def read_file(fname):
    if not isinstance(fname, path_type):
        fname = Path(fname).resolve()

    extension = fname.suffix.lower()
    if extension in [".gzip", ".gz", ".bz2", ".zip"]:
        contents = _open_compressed(fname)
    elif extension in [".cnv", ".edf", ".txt", ".ros", ".btl"]:
        contents = fname.read_bytes()
    else:
        raise ValueError(
            "Unrecognized file extension. Expected .cnv, .edf, .txt, .ros, or .btl got {}".format(
                extension
            )
        )
    # Read as bytes but we need toreturn strings for the parsers.
    text = contents.decode(encoding="utf-8", errors="replace")
    return StringIO(text)


def alphanum_key(s):
    """Order files in a 'human' expected fashion."""
    key = re.split(r"(\d+)", s)
    key[1::2] = list(map(int, key[1::2]))
    return key
