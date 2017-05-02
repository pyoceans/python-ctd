from __future__ import (absolute_import, division, print_function)

__version__ = '0.3.0'

from pandas import DataFrame, Index, Series

from .ctd import asof, from_cnv, from_edf, from_fsi, rosette_summary
from .plotting import (
    extrap_sec,
    gen_topomask,
    get_maxdepth,
    plot,
    plot_section,
    plot_vars,
)
from .processing import (
    barrier_layer_thickness,
    bindata,
    cell_thermal_mass,
    derive_cnv,
    despike,
    lp_filter,
    mixed_layer_depth,
    movingaverage,
    press_check,
    smooth,
    split,
)

__all__ = [
    asof,
    barrier_layer_thickness,
    bindata,
    cell_thermal_mass,
    derive_cnv,
    despike,
    extrap_sec,
    from_cnv,
    from_edf,
    from_fsi,
    gen_topomask,
    get_maxdepth,
    lp_filter,
    mixed_layer_depth,
    movingaverage,
    plot,
    plot_section,
    plot_vars,
    press_check,
    rosette_summary,
    smooth,
    split,
]

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
