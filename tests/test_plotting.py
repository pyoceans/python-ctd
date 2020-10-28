from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ctd


matplotlib.use("Agg")

data_path = Path(__file__).parent.joinpath("data")


def _assert_is_valid_plot_return_object(objs):
    if isinstance(objs, np.ndarray):
        for el in objs.flat:
            assert isinstance(el, plt.Axes), (
                "one of 'objs' is not a "
                "matplotlib Axes instance, "
                "type encountered {0!r}"
                "".format(el.__class__.__name__)
            )
    else:
        assert isinstance(objs, (plt.Artist, tuple, dict)), (
            "objs is neither an ndarray of Artist instances nor a "
            'single Artist instance, tuple, or dict, "objs" is a {0!r} '
            "".format(objs.__class__.__name__)
        )


def _check_plot_works(f, *args, **kwargs):
    ax = f(*args, **kwargs)

    _assert_is_valid_plot_return_object(ax)
    plt.close()


# BasicPlotting.
@pytest.fixture
def xbt():
    yield ctd.from_edf(data_path.joinpath("XBT.EDF.zip"))
    plt.close("all")


@pytest.fixture
def fsi():
    yield ctd.from_fsi(data_path.joinpath("FSI.txt.gz"), skiprows=9)
    plt.close("all")


@pytest.fixture
def cnv():
    yield ctd.from_cnv(data_path.joinpath("small.cnv.bz2"))
    plt.close("all")


def test_xbt_plot(xbt):
    _check_plot_works(xbt["temperature"].plot_cast)


def test_cnv_temperature(cnv):
    _check_plot_works(cnv["t090C"].plot_cast)
