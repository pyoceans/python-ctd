from pandas_flavor import register_dataframe_accessor

from .read import from_bl, from_btl, from_cnv, from_edf, from_fsi


@register_dataframe_accessor("ctd")
class CTD(object):
    def __init__(self, data):
        self._data

    def read_bl(fname):
        return from_bl(fname)

    def read_btl(fname):
        return from_btl(fname)

    def read_edf(fname):
        return from_edf(fname)

    def read_cnv(fname):
        return from_cnv(fname)

    def read_fsi(fname, skiprows=9):
        return from_fsi(fname, skiprows=skiprows)
