"""Test reading functionality."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ctd
from ctd.read import _read_file

data_path = Path(__file__).parent.joinpath("data")


# Test `_read_file` and `_open_compressed`.
def test_zip():
    """Test reading from zip."""
    cfile = _read_file(data_path.joinpath("XBT.EDF.zip"))
    assert isinstance(cfile, io.StringIO)


def test_gzip():
    """Test reading from gzip."""
    cfile = _read_file(data_path.joinpath("XBT.EDF.gz"))
    assert isinstance(cfile, io.StringIO)


def test_bz2():
    """Test reading from bzip2."""
    cfile = _read_file(data_path.joinpath("XBT.EDF.bz2"))
    assert isinstance(cfile, io.StringIO)


def test_uncompresed():
    """Test reading from uncompressed file."""
    cfile = _read_file(data_path.joinpath("XBT.EDF"))
    assert isinstance(cfile, io.StringIO)


# Test ctd DataFrame.
@pytest.fixture
def xbt():
    """Load zip EDF file."""
    return ctd.from_edf(data_path.joinpath("XBT.EDF.zip"))


@pytest.fixture
def fsi():
    """Load gzip FSI file."""
    return ctd.from_fsi(data_path.joinpath("FSI.txt.gz"), skiprows=9)


@pytest.fixture
def cnv():
    """Load bzip2 CNV file."""
    return ctd.from_cnv(data_path.joinpath("small.cnv.bz2"))


@pytest.fixture
def btl():
    """Load uncompressed BTL file."""
    return ctd.from_btl(data_path.joinpath("btl", "bottletest.btl"))


@pytest.fixture
def btl_as_stream():
    """Load stream BTL data."""
    with Path.open(
        data_path.joinpath("btl", "alt_bottletest.BTL"),
        mode="rb",
    ) as f:
        stream = io.StringIO(f.read().decode("cp1252"))
    return ctd.from_btl(stream)


@pytest.fixture
def ros():
    """Load uncompressed ROS file."""
    return ctd.rosette_summary(data_path.joinpath("CTD", "g01l03s01m-m2.ros"))


def test_xbt_is_dataframe(xbt):
    """Test XBT."""
    assert isinstance(xbt, pd.DataFrame)
    assert not xbt.empty


def test_fsi_is_dataframe(fsi):
    """Test FSI."""
    assert isinstance(fsi, pd.DataFrame)
    assert not fsi.empty


def test_cnv_is_dataframe(cnv):
    """Test CNV."""
    assert isinstance(cnv, pd.DataFrame)
    assert not cnv.empty


def test_btl_is_dataframe(btl):
    """Test BTL."""
    assert isinstance(btl, pd.DataFrame)
    assert not btl.empty


def test_btl_with_dup_cols(btl_as_stream):
    """Test BTL with duplicated columns."""
    assert all(col in btl_as_stream.columns for col in ["Bottle", "Bottle_"])


def test_btl_as_stringio(btl_as_stream):
    """Test BTL from stream."""
    assert isinstance(btl_as_stream, pd.DataFrame)
    assert not btl_as_stream.empty


def test_ros_is_dataframe(ros):
    """Test ROS."""
    assert isinstance(ros, pd.DataFrame)
    assert not ros.empty


def test_ros_no_file_name():
    """Test is if missing the 'File Name' is set to 'unknown'."""
    with Path.open(
        data_path.joinpath("CTD", "fixstation_hl_02.ros"),
        mode="rb",
    ) as f:
        stream = io.StringIO(f.read().decode("cp1252"))
    data = ctd.rosette_summary(stream)
    assert data._metadata["name"] == "unknown"  # noqa: SLF001


def test_header_parse():
    """Test header parsing."""
    # File with missing positions.
    xbt = ctd.from_edf(data_path.joinpath("C3_00005.edf"))
    assert xbt._metadata["lon"] is None  # noqa: SLF001
    assert xbt._metadata["lat"] is None  # noqa: SLF001

    # File with valid positions.
    xbt = ctd.from_edf(data_path.joinpath("XBT.EDF"))
    np.testing.assert_almost_equal(xbt._metadata["lon"], -39.8790283)  # noqa: SLF001
    np.testing.assert_almost_equal(xbt._metadata["lat"], -19.7174805)  # noqa: SLF001


def test_header_parse_blank_line():
    """Check if file is loaded when the header section contains blank lines.

    If the blank line in the header causes to exit before reading it,
    the line looking for the Date in the `from_btl` will throw a ValueError.
    """
    btl = ctd.from_btl(
        data_path.joinpath(
            "btl",
            "blank_line_header.btl",
        ),
    )

    assert btl._metadata["names"].index("Date")  # noqa: SLF001


def test_pressure_field_labels():
    """Support different pressure field labels encountered in CNV files."""
    for fname in sorted(data_path.glob("press-pass*.cnv")):
        ctd.from_cnv(fname)
    for fname in sorted(data_path.glob("press-fails*.cnv")):
        with pytest.raises(
            ValueError,
            match="Expected one pressure/depth column, didn't receive any",
        ):
            ctd.from_cnv(fname)
