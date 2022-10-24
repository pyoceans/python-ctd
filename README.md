# python-ctd

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11396.svg)](https://doi.org/10.5281/zenodo.11396)
[![Tests](https://github.com/pyoceans/python-ctd/actions/workflows/tests.yml/badge.svg)](https://github.com/pyoceans/python-ctd/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/ctd.svg?style=plastic)](https://pypi.python.org/pypi/ctd)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat)](https://github.com/pyoceans/python-ctd/blob/main/LICENSE.txt)

Tools to load hydrographic data as pandas DataFrame with some handy methods for
data pre-processing and analysis

This module can load [SeaBird CTD (CNV)](https://www.seabird.com/),
[Sippican XBT (EDF)](https://www.lockheedmartin.com/en-us/products/oceanographic-instrumentation.html),
and [Falmouth CTD (ASCII)](https://www.falmouth.com/) formats.

## Quick intro

You can install the CTD package with

```shell
conda install ctd --channel conda-forge
```

or

```shell
pip install ctd
```


and then,

```python
from pathlib import Path
import ctd

path = Path('tests', 'data', 'CTD')
fname = path.joinpath('g01l06s01.cnv.gz')

down, up = ctd.from_cnv(fname).split()
ax = down['t090C'].plot_cast()
```

![Bad Processing](https://raw.githubusercontent.com/pyoceans/python-ctd/main/docs/readme_01.png)

We can do [better](https://www.go-ship.org/Manual/McTaggart_et_al_CTD.pdf):

```python
temperature = down['t090C']

fig, ax = plt.subplots(figsize=(5.5, 6))
temperature.plot_cast(ax=ax)
temperature.remove_above_water()\
           .despike()\
           .lp_filter()\
           .press_check()\
           .interpolate(method='index',
                        limit_direction='both',
                        limit_area='inside')\
           .bindata(delta=1, method='interpolate')\
           .smooth(window_len=21, window='hanning') \
           .plot_cast(ax=ax)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel('Temperature (°C)')
```

![Good Processing](https://raw.githubusercontent.com/pyoceans/python-ctd/main/docs/readme_02.png)

## Try it out on mybinder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyoceans/python-ctd/HEAD?labpath=notebooks)
