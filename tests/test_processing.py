import numpy as np
import pandas as pd
import pytest

import ctd  # noqa


@pytest.fixture
def series():
    index = np.r_[np.linspace(-5, 10, 20), np.linspace(10, -5, 20)]
    yield pd.Series(data=np.arange(len(index)), index=index)


@pytest.fixture
def df():
    index = np.r_[np.linspace(-5, 10, 20), np.linspace(10, -5, 20)]
    yield pd.DataFrame(data=np.arange(len(index)), index=index)


def test_remove_above_water_series(series):
    assert any(series.index < 0)
    assert not any(series.remove_above_water().index < 0)


def test_remove_above_water_df(df):
    assert any(df.index < 0)
    assert not any(df.remove_above_water().index < 0)


def test_split_series(series):
    splitted = series.split()
    down, up = splitted
    assert isinstance(splitted, tuple)
    assert series.equals(pd.concat([down, up[::-1]]))


def test_split_df(df):
    splitted = df.split()
    down, up = splitted
    assert isinstance(splitted, tuple)
    assert df.equals(pd.concat([down, up[::-1]]))


def test_press_check_series(series):
    # reverse 7th and 9th and confirm they are removed after the `press_check`.
    index = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10]
    series = pd.Series(data=np.random.randn(len(index)), index=index)
    series = series.press_check()
    assert np.isnan(series.iloc[7])
    assert np.isnan(series.iloc[9])


def test_press_check_df(df):
    # reverse 7th and 9th and confirm they are removed after the `press_check`.
    index = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10]
    arr = np.random.randn(len(index))
    df = pd.DataFrame(data=np.c_[arr, arr], index=index)
    df = df.press_check()
    assert np.isnan(df.iloc[7]).all()
    assert np.isnan(df.iloc[9]).all()


def test_bindata_average(series):
    delta = 1.0
    index = series.remove_above_water().split()[0].bindata(delta=delta).index
    assert all(index.values == np.arange(1, 9, delta) + delta / 2)
    assert np.unique(np.diff(index.values)) == delta

    delta = 2
    index = series.remove_above_water().split()[0].bindata(delta=delta).index
    assert all(index.values == np.arange(1, 9, delta) + delta / 2)
    assert np.unique(np.diff(index.values)) == delta
