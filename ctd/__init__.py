# -*- coding: utf-8 -*-

__version__ = '0.1.0'

from pandas import Index, Series, DataFrame

from ctd import asof, from_edf, from_cnv, from_fsi, rosette_summary
from processing import (data_conversion, align, despike, seabird_filter,
                        cell_thermal_mass, press_check, bindata, split,
                        pmel_inversion_check, smooth, mixed_layer_depth,
                        barrier_layer_thickness)
from plotting import (get_maxdepth, extrap_sec, gen_topomask, plot, plot_vars,
                      plot_section)


# Attach methods.
Index.asof = asof

Series.plot = plot
Series.split = split
Series.smooth = smooth
Series.despike = despike
Series.bindata = bindata
Series.press_check = press_check

DataFrame.split = split
DataFrame.from_cnv = staticmethod(from_cnv)
DataFrame.from_edf = staticmethod(from_edf)
DataFrame.from_fsi = staticmethod(from_fsi)
DataFrame.plot_vars = plot_vars
DataFrame.press_check = press_check
DataFrame.get_maxdepth = get_maxdepth
DataFrame.plot_section = plot_section
