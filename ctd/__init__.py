from ._version import get_versions  # noqa
from .core import CTD as _CTD  # noqa
from .plotting import plot_cast
from .processing import (
    bindata,
    despike,
    lp_filter,
    movingaverage,
    press_check,
    remove_above_water,
    smooth,
    split,
)
from .read import (
    from_bl,
    from_btl,
    from_cnv,
    from_edf,
    from_fsi,
    rosette_summary,
)

__version__ = get_versions()["version"]
del get_versions


__all__ = [
    "bindata",
    "despike",
    "from_bl",
    "from_btl",
    "from_cnv",
    "from_edf",
    "from_fsi",
    "lp_filter",
    "movingaverage",
    "plot_cast",
    "press_check",
    "remove_above_water",
    "rosette_summary",
    "smooth",
    "split",
]
