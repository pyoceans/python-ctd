from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from ctd import DataFrame, derive_cnv, lp_filter
from ctd.utilities import Path

data_path = Path(__file__).parent.joinpath("data")


@pytest.fixture
def load_spiked_ctd(name):
    return DataFrame.from_cnv(data_path.joinpath(name))


# Split.
def test_split_return_tuple():
    raw = load_spiked_ctd("CTD-spiked-unfiltered.cnv.bz2").split()
    assert isinstance(raw, tuple)


def test_split_cnv():
    cast = load_spiked_ctd("CTD-spiked-unfiltered.cnv.bz2")
    downcast, upcast = cast.split()
    assert downcast.index.size + upcast.index.size == cast.index.size


# Despike.
def test_despike():
    # Looking at downcast only.
    dirty = load_spiked_ctd("CTD-spiked-filtered.cnv.bz2")["c0S/m"].split()[0]
    clean = dirty.despike(n1=2, n2=20, block=500)
    spikes = clean.isnull()
    equal = (dirty[~spikes] == clean[~spikes]).all()
    assert spikes.any() and equal


# Filter.
def test_lp_filter():
    kw = {"sample_rate": 24.0, "time_constant": 0.15}
    expected = load_spiked_ctd("CTD-spiked-filtered.cnv.bz2").index.values
    unfiltered = load_spiked_ctd("CTD-spiked-unfiltered.cnv.bz2").index.values
    filtered = lp_filter(unfiltered, **kw)
    # Caveat: Not really a good test...
    np.testing.assert_almost_equal(filtered, expected, decimal=1)


# Pressure check.
def test_press_check():
    unchecked = load_spiked_ctd("CTD-spiked-unfiltered.cnv.bz2")["t090C"]
    press_checked = unchecked.press_check()
    reversals = press_checked.isnull()
    equal = (unchecked[~reversals] == press_checked[~reversals]).all()
    assert reversals.any() and equal


def test_bindata():
    delta = 1.0
    down = load_spiked_ctd("CTD-spiked-filtered.cnv.bz2")["t090C"].split()[0]
    down = down.bindata(delta=delta)
    assert np.unique(np.diff(down.index.values)) == delta


def test_derive_cnv():
    cast = load_spiked_ctd("CTD-spiked-unfiltered.cnv.bz2")
    cast.lat = cast["latitude"].mean()
    cast.lon = cast["longitude"].mean()
    derived = derive_cnv(cast)
    new_cols = set(derived).symmetric_difference(cast.columns)
    assert ["CT", "SA", "SP", "SR", "sigma0_CT", "z"] == sorted(new_cols)
