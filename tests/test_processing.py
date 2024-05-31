"""Test processing methods."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def series():
    """Load data series."""
    index = np.r_[np.linspace(-5, 10, 20), np.linspace(10, -5, 20)]
    return pd.Series(data=np.arange(len(index)), index=index)


@pytest.fixture()
def df():
    """Load data frame."""
    index = np.r_[np.linspace(-5, 10, 20), np.linspace(10, -5, 20)]
    return pd.DataFrame(data=np.arange(len(index)), index=index)


def test_remove_above_water_series(series):
    """Test remove above water series."""
    assert any(series.index < 0)
    assert not any(series.remove_above_water().index < 0)


def test_remove_above_water_df(df):
    """Test remove above water dataframe."""
    assert any(df.index < 0)
    assert not any(df.remove_above_water().index < 0)


def test_remove_up_to_series(series):
    """Test remove up to series."""
    idx = 10
    assert any(series.index < idx)
    assert not any(series.remove_up_to(idx=idx).index < idx)


def test_remove_up_to_df(df):
    """Test remove up dataframe."""
    idx = 10
    assert any(df.index < idx)
    assert not any(df.remove_up_to(idx=idx).index < idx)


def test_split_series(series):
    """Test split series."""
    split = series.split()
    down, up = split
    assert isinstance(split, tuple)
    assert series.equals(pd.concat([down, up[::-1]]))


def test_split_df(df):
    """Test split dataframe."""
    split = df.split()
    down, up = split
    assert isinstance(split, tuple)
    assert df.equals(pd.concat([down, up[::-1]]))


def test_press_check_series(series):
    """Test pressure check series.

    Reverse 7th and 9th and confirm they are removed after the `press_check`.
    """
    index = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10]
    rng = np.random.default_rng()
    series = pd.Series(data=rng.standard_normal(len(index)), index=index)
    series = series.press_check()
    assert np.isnan(series.iloc[7])
    assert np.isnan(series.iloc[9])


def test_press_check_df(df):
    """Test pressure check dataframe.

    Reverse 7th and 9th and confirm they are removed after the `press_check`.
    """
    index = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10]
    rng = np.random.default_rng()
    arr = rng.standard_normal(len(index))
    df = pd.DataFrame(data=np.c_[arr, arr], index=index)
    df = df.press_check()
    assert np.isnan(df.iloc[7]).all()
    assert np.isnan(df.iloc[9]).all()


def test_bindata_average(series):
    """Test bin data."""
    delta = 1.0
    index = series.remove_above_water().split()[0].bindata(delta=delta).index
    assert all(index.to_numpy() == np.arange(1, 9, delta) + delta / 2)
    assert np.unique(np.diff(index.to_numpy())) == delta

    delta = 2
    index = series.remove_above_water().split()[0].bindata(delta=delta).index
    assert all(index.to_numpy() == np.arange(1, 9, delta) + delta / 2)
    assert np.unique(np.diff(index.to_numpy())) == delta
