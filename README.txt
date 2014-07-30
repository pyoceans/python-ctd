python-ctd
==========

.. image:: https://badge.fury.io/py/ctd.png
   :target: http://badge.fury.io/py/ctd
   :alt: Latest version
.. image:: https://api.travis-ci.org/ocefpaf/python-ctd.png?branch=master
   :target: https://travis-ci.org/ocefpaf/python-ctd
   :alt: Travs-CI
.. image:: http://bottlepy.org/docs/dev/_static/Gittip.png
   :target: https://www.gittip.com/ocefpaf/
   :alt: Gittip

Tools to load hydrographic data into pandas DataFrame (and some
rudimentary methods for data pre-processing/analysis).

This module can load `SeaBird CTD
(CNV) <http://www.seabird.com/software/SBEDataProcforWindows.htm>`_,
`Sippican XBT (EDF) <http://www.sippican.com/>`_, and `Falmouth CTD
(ASCII) <http://www.falmouth.com/>`_ formats.

Quick intro
-----------

.. code-block:: bash

     pip install ctd

and then,

.. code-block:: python

    kw = dict(compression='gzip')
    fname = './test/data/CTD/g01l06s01.cnv.gz'
    cast = DataFrame.from_cnv(fname, **kw)
    downcast, upcast = cast.split()
    fig, ax = downcast['t090C'].plot()

.. image:: https://raw.githubusercontent.com/ocefpaf/python-ctd/master/docs/readme_01.png
   :alt: Bad Processing

We can do
`better <http://www.go-ship.org/Manual/McTaggart_et_al_CTD.pdf>`_:

.. code-block:: python

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

.. image:: https://raw.githubusercontent.com/ocefpaf/python-ctd/master/docs/readme_02.png
   :alt: Good Processing

Not so quick intro
------------------

`Profiles <http://ocefpaf.github.io/python4oceanographers/blog/2013/05/27/CTD2DataFrame/>`_ and
`sections <http://ocefpaf.github.io/python4oceanographers/blog/2013/07/29/python-ctd/>`_.

Author
------

Filipe Fernandes
