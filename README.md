# python-ctd

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11396.svg)](https://doi.org/10.5281/zenodo.11396) [![Build Status](https://travis-ci.org/pyoceans/python-ctd.svg?branch=master)](https://travis-ci.org/pyoceans/python-ctd) [![PyPI](https://img.shields.io/pypi/v/ctd.svg?style=plastic)](https://pypi.python.org/pypi/ctd) [![Build status](https://ci.appveyor.com/api/projects/status/m1wxtsb8gpm96i53?svg=true)](https://ci.appveyor.com/project/ocefpaf/python-ctd) [![License](http://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat)](https://github.com/pyoceans/python-ctd/blob/master/LICENSE.txt)

Tools to load hydrographic data as pandas DataFrame with some handy methods for
data pre-processing and analysis

This module can load [SeaBird CTD (CNV)](http://www.seabird.com/software/SBEDataProcforWindows.htm),
[Sippican XBT (EDF)](http://www.sippican.com/),
and [Falmouth CTD (ASCII)](http://www.falmouth.com/) formats.

## Quick intro

```shell
conda install ctd --channel conda-forge
```

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

![Bad Processing](https://raw.githubusercontent.com/pyoceans/python-ctd/master/docs/readme_01.png)

We can do [better](http://www.go-ship.org/Manual/McTaggart_et_al_CTD.pdf):

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

![Good Processing](https://raw.githubusercontent.com/pyoceans/python-ctd/master/docs/readme_02.png)
