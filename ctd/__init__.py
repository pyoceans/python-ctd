from __future__ import absolute_import, division, print_function
import warnings

from pandas import DataFrame, Index, Series

from .ctd import (
    asof,
    from_btl,
    from_cnv,
    from_edf,
    from_fsi,
    rosette_summary
)
try:
    from .plotting import (
        extrap_sec,
        gen_topomask,
        get_maxdepth,
        plot,
        plot_section,
        plot_vars,
    )
    HAS_MATPLOTLIB = True
except RuntimeError:
    warnings.warn("""
Plotting routines not accesible, probably due to problems with matplotlib. See 
https://matplotlib.org/faq/osx_framework.html if you use MacOS and annaconda
""")
    HAS_MATPLOTLIB = False
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


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

if HAS_MATPLOTLIB:
    plotting_functions = [
        extrap_sec,
        gen_topomask,
        get_maxdepth,
        plot,
        plot_section,
        plot_vars
    ]
else:
    plotting_functions = []

__all__ = plotting_functions + [
    asof,
    barrier_layer_thickness,
    bindata,
    cell_thermal_mass,
    derive_cnv,
    despike,
    from_cnv,
    from_btl,
    from_edf,
    from_fsi,
    lp_filter,
    mixed_layer_depth,
    movingaverage,
    press_check,
    rosette_summary,
    smooth,
    split,
]

# Attach methods.
Index.asof = asof

Series.split = split
Series.smooth = smooth
Series.despike = despike
Series.bindata = bindata
Series.press_check = press_check
if HAS_MATPLOTLIB:
    Series.plot = plot

DataFrame.split = split
DataFrame.from_cnv = staticmethod(from_cnv)
DataFrame.from_btl = staticmethod(from_btl)
DataFrame.from_edf = staticmethod(from_edf)
DataFrame.from_fsi = staticmethod(from_fsi)
DataFrame.press_check = press_check
if HAS_MATPLOTLIB:
    DataFrame.plot_vars = plot_vars
    DataFrame.get_maxdepth = get_maxdepth
    DataFrame.plot_section = plot_section
