from pathlib import Path

import numpy as np
import pytest

import ctd

data_path = Path(__file__).parent.joinpath("data")


@pytest.fixture
def spiked_ctd():
    yield ctd.from_cnv(data_path.joinpath("CTD-spiked-unfiltered.cnv.bz2"))


@pytest.fixture
def filtered_ctd():
    yield ctd.from_cnv(data_path.joinpath("CTD-spiked-filtered.cnv.bz2"))


def test_despike_real_data(filtered_ctd):
    # Looking at downcast only.
    dirty = filtered_ctd["c0S/m"].split()[0]
    clean = dirty.despike(n1=2, n2=20, block=500)
    spikes = clean.isnull()
    equal = (dirty[~spikes] == clean[~spikes]).all()
    assert spikes.any() and equal


def test_lp_filter_real_data(spiked_ctd, filtered_ctd):
    kw = {"sample_rate": 24.0, "time_constant": 0.15}
    expected = filtered_ctd.index.values
    filtered = spiked_ctd.lp_filter(**kw).index
    # Caveat: Not really a good test...
    np.testing.assert_almost_equal(filtered, expected, decimal=1)


def test_press_check_real_data(spiked_ctd):
    unchecked = spiked_ctd["t090C"]
    press_checked = unchecked.press_check()
    reversals = press_checked.isnull()
    equal = (unchecked[~reversals] == press_checked[~reversals]).all()
    assert reversals.any() and equal


def test_processing_chain_spiked_ctd(spiked_ctd):
    down, up = spiked_ctd.remove_above_water().split()
    temp = down["t090C"]  # despike is a series only method
    temp = (
        temp.despike()
        .lp_filter(sample_rate=24.0, time_constant=0.15)
        .press_check()
        .bindata()
        .smooth(window_len=21, window="hanning")
    )
    assert all(spiked_ctd.columns == down.columns)
