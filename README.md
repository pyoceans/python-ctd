# python-ctd

[![PyPI](https://badge.fury.io/py/ctd.png)](http://badge.fury.io/py/ctd)
[![Build](https://api.travis-ci.org/ocefpaf/python-ctd.png?branch=master)](https://travis-ci.org/ocefpaf/python-ctd)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.11396.png)](http://dx.doi.org/10.5281/zenodo.11396)

[![Gittip](http://bottlepy.org/docs/dev/_static/Gittip.png)](https://www.gittip.com/ocefpaf/)

Tools to load hydrographic data into pandas DataFrame (and some rudimentary methods for data pre-processing/analysis).

This module can load [SeaBird CTD (CNV)][SBE], [Sippican XBT (EDF)][XBT],
and [Falmouth CTD (ASCII)][FSI] formats.

[SBE]: http://www.seabird.com/software/SBEDataProcforWindows.htm

[XBT]: http://www.sippican.com/

[FSI]: http://www.falmouth.com/

## Quick intro
```bash
 pip install ctd
```

and then,

```python
from ctd import DataFrame
kw = dict(compression='gzip')
fname = './test/data/CTD/g01l06s01.cnv.gz'
cast = DataFrame.from_cnv(fname, **kw)
downcast, upcast = cast.split()
fig, ax = downcast['t090C'].plot()
```

![Bad Processing](https://raw.githubusercontent.com/ocefpaf/python-ctd/master/docs/readme_01.png)

We can do [better](http://www.go-ship.org/Manual/McTaggart_et_al_CTD.pdf):

```python
from ctd import DataFrame, lp_filter, movingaverage
kw.update(below_water=True)
cast = DataFrame.from_cnv(fname, **kw)
downcast, upcast = cast.split()
temperature = downcast['t090C'].despike(n1=2, n2=20, block=100)
temperature.index = lp_filter(temperature.index.values)
temperature = temperature.bindata(delta=1)
temperature = temperature.interpolate()
temperature = temperature.smooth(window_len=21, window='hanning')
fig, ax = temperature.plot()
ax.axis([0, 30, 2000, 0])
ax.set_ylabel("Pressure [dbar]")
ax.set_xlabel(u'Temperature [\u00b0C]')
```

![Good Processing](https://raw.githubusercontent.com/ocefpaf/python-ctd/master/docs/readme_02.png)


## Not so quick intro
[Profiles](http://ocefpaf.github.io/python4oceanographers/blog/2013/05/27/CTD2DataFrame/)
and [sections](http://ocefpaf.github.io/python4oceanographers/blog/2013/07/29/python-ctd/).

Author
------
Filipe Fernandes
