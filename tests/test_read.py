from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ctd
from ctd.read import _read_file

data_path = Path(__file__).parent.joinpath("data")


# Test `_read_file` and `_open_compressed`.
def test_zip():
    cfile = _read_file(data_path.joinpath("XBT.EDF.zip"))
    assert isinstance(cfile, StringIO)


def test_gzip():
    cfile = _read_file(data_path.joinpath("XBT.EDF.gz"))
    assert isinstance(cfile, StringIO)


def test_bz2():
    cfile = _read_file(data_path.joinpath("XBT.EDF.bz2"))
    assert isinstance(cfile, StringIO)


def test_uncompresed():
    cfile = _read_file(data_path.joinpath("XBT.EDF"))
    assert isinstance(cfile, StringIO)


# Test ctd DataFrame.
@pytest.fixture
def xbt():
    yield ctd.from_edf(data_path.joinpath("XBT.EDF.zip"))


@pytest.fixture
def fsi():
    yield ctd.from_fsi(data_path.joinpath("FSI.txt.gz"), skiprows=9)


@pytest.fixture
def cnv():
    yield ctd.from_cnv(data_path.joinpath("small.cnv.bz2"))


@pytest.fixture
def btl():
    yield ctd.from_btl(data_path.joinpath("btl", "bottletest.btl"))


@pytest.fixture
def btl_as_stream():
    file = open(mode="rb", file=data_path.joinpath("btl", "alt_bottletest.BTL"))
    stream = StringIO(file.read().decode("cp1252"))
    yield ctd.from_btl(stream)


@pytest.fixture
def ros():
    yield ctd.rosette_summary(data_path.joinpath("CTD", "g01l03s01m-m2.ros"))


def test_xbt_is_dataframe(xbt):
    assert isinstance(xbt, pd.DataFrame)
    assert not xbt.empty


def test_fsi_is_dataframe(fsi):
    assert isinstance(fsi, pd.DataFrame)
    assert not fsi.empty


def test_cnv_is_dataframe(cnv):
    assert isinstance(cnv, pd.DataFrame)
    assert not cnv.empty


def test_btl_is_dataframe(btl):
    assert isinstance(btl, pd.DataFrame)
    assert not btl.empty


def test_btl_with_dup_cols(btl_as_stream):
    assert all(col in btl_as_stream.columns for col in ["Bottle", "Bottle_"])


def test_btl_as_stringio(btl_as_stream):
    assert isinstance(btl_as_stream, pd.DataFrame)
    assert not btl_as_stream.empty


def test_ros_is_dataframe(ros):
    assert isinstance(ros, pd.DataFrame)
    assert not ros.empty


# HeaderTest.
def test_header_parse():
    # file with missing positions
    xbt = ctd.from_edf(data_path.joinpath("C3_00005.edf"))
    assert xbt._metadata["lon"] is None
    assert xbt._metadata["lat"] is None

    # file with valid positions
    xbt = ctd.from_edf(data_path.joinpath("XBT.EDF"))
    np.testing.assert_almost_equal(xbt._metadata["lon"], -39.8790283)
    np.testing.assert_almost_equal(xbt._metadata["lat"], -19.7174805)


def test_header_parse_blank_line():
    # check that a BTL file can still be loaded if the header section contains blank lines

    # if the blank line in the header causes the reader to exit before reading the file
    # the line looking for the Date in the ctd.from_btl() will throw a ValueError.
    btl = ctd.from_btl(
        data_path.joinpath(
            "btl",
            "blank_line_header.btl",
        ),
    )

    # if a value error wasn't thrown, ensure the names array for the _metadata was set
    assert btl._metadata["names"].index("Date")


def test_pressure_field_labels():
    """
    Support different pressure field labels encountered in Sea-Bird cnv files.

    """
    for fname in sorted(data_path.glob("press-pass*.cnv")):
        ctd.from_cnv(fname)
    for fname in sorted(data_path.glob("press-fails*.cnv")):
        with pytest.raises(ValueError):
            ctd.from_cnv(fname)
