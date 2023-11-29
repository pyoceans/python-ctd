"""
Read module
"""

import bz2
import collections
import gzip
import linecache
import re
import warnings
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import gsw
import numpy as np
import pandas as pd


def _basename(fname):
    """Return file name without path."""
    if not isinstance(fname, Path):
        fname = Path(fname)
    path, name, ext = fname.parent, fname.stem, fname.suffix
    return path, name, ext


def _normalize_names(name):
    """Normalize column names."""
    name = name.strip()
    name = name.strip("*")
    return name


def _open_compressed(fname):
    """Open compressed gzip, gz, zip or bz2 files."""
    extension = fname.suffix.casefold()
    if extension in [".gzip", ".gz"]:
        cfile = gzip.open(str(fname))
    elif extension == ".bz2":
        cfile = bz2.BZ2File(str(fname))
    elif extension == ".zip":
        # NOTE: Zip format may contain more than one file in the archive
        # (similar to tar), here we assume that there is just one file per
        # zipfile!  Also, we ask for the name because it can be different from
        # the zipfile file!!
        zfile = zipfile.ZipFile(str(fname))
        name = zfile.namelist()[0]
        cfile = zfile.open(name)
    else:
        raise ValueError(
            "Unrecognized file extension. Expected .gzip, .bz2, or .zip, got {}".format(
                extension,
            ),
        )
    contents = cfile.read()
    cfile.close()
    return contents


def _read_file(fname):
    """Read file contents, or read from StringIO object."""
    if isinstance(fname, StringIO):
        fname.seek(0)
        text = fname.read()
        return StringIO(text)

    if not isinstance(fname, Path):
        fname = Path(fname).resolve()

    extension = fname.suffix.casefold()
    if extension in [".gzip", ".gz", ".bz2", ".zip"]:
        contents = _open_compressed(fname)
    elif extension in [".cnv", ".edf", ".txt", ".ros", ".btl"]:
        contents = fname.read_bytes()
    else:
        raise ValueError(
            f"Unrecognized file extension. Expected .cnv, .edf, .txt, .ros, or .btl got {extension}",
        )
    # Read as bytes but we need to return strings for the parsers.
    text = contents.decode(encoding="utf-8", errors="replace")
    return StringIO(text)


def _remane_duplicate_columns(names):
    """Rename a column when it is duplicated."""
    items = collections.Counter(names).items()
    dup = []
    for item, count in items:
        if count > 2:
            raise ValueError(
                f"Cannot handle more than two duplicated columns. Found {count} for {item}.",
            )
        if count > 1:
            dup.append(item)

    # since we can assume there are only two instances of a word in the list, how about we find the last
    # index of an instance, which will be the second occurrence of the item
    second_occurrences = [len(names) - names[::-1].index(item) - 1 for item in dup]
    for idx in second_occurrences:
        names[idx] = f"{names[idx]}_"
    return names


def _parse_seabird(lines, ftype):
    """Parse searbird formats."""
    # Initialize variables.
    lon = lat = time = None, None, None
    fname = None
    skiprows = 0

    metadata = {}
    header, config, names = [], [], []
    for k, line in enumerate(lines):
        line = line.strip()

        # Only cnv has columns names, for bottle files we will use the variable row.
        if ftype == "cnv":
            if "# name" in line:
                name, unit = line.split("=")[1].split(":")
                name, unit = list(map(_normalize_names, (name, unit)))
                names.append(name)

        # Seabird headers starts with *.
        if line.startswith("*"):
            header.append(line)
            if "FileName" in line:
                file_path = line.split("=")[-1].strip()
                fname = Path(file_path).stem

        # Seabird configuration starts with #.
        if line.startswith("#"):
            config.append(line)

        # NMEA position and time.
        if "NMEA Latitude" in line:
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split("=")[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == "S":
                lat = -(lat[0] + lat[1] / 60.0)
            elif hemisphere == "N":
                lat = lat[0] + lat[1] / 60.0
            else:
                raise ValueError("Latitude not recognized.")
        if "NMEA Longitude" in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split("=")[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == "W":
                lon = -(lon[0] + lon[1] / 60.0)
            elif hemisphere == "E":
                lon = lon[0] + lon[1] / 60.0
            else:
                raise ValueError("Latitude not recognized.")
        if "NMEA UTC (Time)" in line:
            time = line.split("=")[-1].strip()
            # Should use some fuzzy datetime parser to make this more robust.
            time = datetime.strptime(time, "%b %d %Y %H:%M:%S")

        # cnv file header ends with *END* while
        if ftype == "cnv":
            if line == "*END*":
                skiprows = k + 1
                break
        else:  # btl.
            # There is no *END* like in a .cnv file, skip two after header info.
            if line != "" and not (line.startswith("*") | line.startswith("#")):
                # Fix commonly occurring problem when Sbeox.* exists in the file
                # the name is concatenated to previous parameter
                # example:
                #   CStarAt0Sbeox0Mm/Kg to CStarAt0 Sbeox0Mm/Kg (really two different params)
                line = re.sub(r"(\S)Sbeox", "\\1 Sbeox", line)

                names = line.split()
                skiprows = k + 2
                break
    if ftype == "btl":
        # Capture stat names column.
        names.append("Statistic")
    metadata.update(
        {
            "name": fname if fname else "unknown",
            "header": "\n".join(header),
            "config": "\n".join(config),
            "names": _remane_duplicate_columns(names),
            "skiprows": skiprows,
            "time": time,
            "lon": lon,
            "lat": lat,
        },
    )
    return metadata


def from_bl(fname):
    """Read Seabird bottle-trip (bl) file

    Example
    -------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> df = ctd.from_bl(str(data_path.joinpath('bl', 'bottletest.bl')))
    >>> df._metadata["time_of_reset"]
    datetime.datetime(2018, 6, 25, 20, 8, 55)

    """
    df = pd.read_csv(
        fname,
        skiprows=2,
        parse_dates=[1],
        index_col=0,
        names=["bottle_number", "time", "startscan", "endscan"],
    )
    df._metadata = {
        "time_of_reset": pd.to_datetime(
            linecache.getline(str(fname), 2)[6:-1],
        ).to_pydatetime(),
    }
    return df


def from_btl(fname):
    """
    DataFrame constructor to open Seabird CTD BTL-ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> bottles = ctd.from_btl(data_path.joinpath('btl', 'bottletest.btl'))

    """
    f = _read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype="btl")

    f.seek(0)

    df = pd.read_fwf(
        f,
        header=None,
        index_col=False,
        names=metadata["names"],
        parse_dates=False,
        skiprows=metadata["skiprows"],
    )
    f.close()

    # At this point the data frame is not correctly lined up (multiple rows
    # for avg, std, min, max or just avg, std, etc).
    # Also needs date,time,and bottle number to be converted to one per line.

    # Get row types, see what you have: avg, std, min, max or just avg, std.
    rowtypes = df[df.columns[-1]].unique()
    # Get times and dates which occur on second line of each bottle.
    date_idx = metadata["names"].index("Date")
    dates = df.iloc[:: len(rowtypes), date_idx].reset_index(drop=True)
    times = df.iloc[1 :: len(rowtypes), date_idx].reset_index(drop=True)
    datetimes = dates + " " + times

    # Fill the Date column with datetimes.
    df.loc[:: len(rowtypes), "Date"] = datetimes.values
    df.loc[1 :: len(rowtypes), "Date"] = datetimes.values

    # Fill missing rows.
    df["Bottle"] = df["Bottle"].fillna(method="ffill")
    df["Date"] = df["Date"].fillna(method="ffill")

    df["Statistic"] = df["Statistic"].str.replace(r"\(|\)", "")  # (avg) to avg

    if "name" not in metadata:
        name = _basename(fname)[1]
        metadata["name"] = str(name)

    dtypes = {
        "bpos": int,
        "pumps": bool,
        "flag": bool,
        "Bottle": int,
        "Scan": int,
        "Statistic": str,
        "Date": str,
    }
    for column in df.columns:
        if column in dtypes:
            df[column] = df[column].astype(dtypes[column])
        else:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                warnings.warn("Could not convert %s to float." % column)

    df["Date"] = pd.to_datetime(df["Date"])
    setattr(df, "_metadata", metadata)
    return df


def from_edf(fname):
    """
    DataFrame constructor to open XBT EDF ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_edf(data_path.joinpath('XBT.EDF.gz'))
    >>> ax = cast['temperature'].plot_cast()

    """
    f = _read_file(fname)
    header, names = [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if line.startswith("Serial Number"):
            serial = line.strip().split(":")[1].strip()
        elif line.startswith("Latitude"):
            try:
                hemisphere = line[-1]
                lat = line.strip(hemisphere).split(":")[1].strip()
                lat = np.float_(lat.split())
                if hemisphere == "S":
                    lat = -(lat[0] + lat[1] / 60.0)
                elif hemisphere == "N":
                    lat = lat[0] + lat[1] / 60.0
            except (IndexError, ValueError):
                lat = None
        elif line.startswith("Longitude"):
            try:
                hemisphere = line[-1]
                lon = line.strip(hemisphere).split(":")[1].strip()
                lon = np.float_(lon.split())
                if hemisphere == "W":
                    lon = -(lon[0] + lon[1] / 60.0)
                elif hemisphere == "E":
                    lon = lon[0] + lon[1] / 60.0
            except (IndexError, ValueError):
                lon = None
        else:
            header.append(line)
            if line.startswith("Field"):
                col, unit = (ln.strip().casefold() for ln in line.split(":"))
                names.append(unit.split()[0])
        if line == "// Data":
            skiprows = k + 1
            break

    f.seek(0)
    df = pd.read_csv(
        f,
        header=None,
        index_col=None,
        names=names,
        skiprows=skiprows,
        delim_whitespace=True,
    )
    f.close()

    df.set_index("depth", drop=True, inplace=True)
    df.index.name = "Depth [m]"
    name = _basename(fname)[1]

    metadata = {
        "lon": lon,
        "lat": lat,
        "name": str(name),
        "header": "\n".join(header),
        "serial": serial,
    }
    setattr(df, "_metadata", metadata)
    return df


def from_cnv(fname):
    """
    DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_cnv(data_path.joinpath('CTD_big.cnv.bz2'))
    >>> downcast, upcast = cast.split()
    >>> ax = downcast['t090C'].plot_cast()

    """
    f = _read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype="cnv")

    f.seek(0)
    df = pd.read_fwf(
        f,
        header=None,
        index_col=None,
        names=metadata["names"],
        skiprows=metadata["skiprows"],
        delim_whitespace=True,
        widths=[11] * len(metadata["names"]),
    )
    f.close()

    prkeys = ["prM ", "prE", "prDM", "pr50M", "pr50M1", "prSM", "prdM", "pr", "depSM"]
    prkey = [key for key in prkeys if key in df.columns]
    if len(prkey) != 1:
        raise ValueError(f"Expected one pressure/depth column, got {prkey}.")
    df.set_index(prkey, drop=True, inplace=True)
    df.index.name = "Pressure [dbar]"
    if prkey == "depSM":
        lat = metadata.get("lat", None)
        if lat is not None:
            df.index = gsw.p_from_z(
                df.index,
                lat,
                geo_strf_dyn_height=0,
                sea_surface_geopotential=0,
            )
        else:
            warnings.war(
                f"Missing latitude information. Cannot compute pressure! Your index is {prkey}, "
                "please compute pressure manually with `gsw.p_from_z` and overwrite your index.",
            )
            df.index.name = prkey

    name = _basename(fname)[1]

    dtypes = {"bpos": int, "pumps": bool, "flag": bool}
    for column in df.columns:
        if column in dtypes:
            df[column] = df[column].astype(dtypes[column])
        else:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                warnings.warn("Could not convert %s to float." % column)

    metadata["name"] = str(name)
    setattr(df, "_metadata", metadata)
    return df


def from_fsi(fname, skiprows=9):
    """
    DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_fsi(data_path.joinpath('FSI.txt.gz'))
    >>> downcast, upcast = cast.split()
    >>> ax = downcast['TEMP'].plot_cast()

    """
    f = _read_file(fname)
    df = pd.read_csv(
        f,
        header="infer",
        index_col=None,
        skiprows=skiprows,
        dtype=float,
        delim_whitespace=True,
    )
    f.close()

    df.set_index("PRES", drop=True, inplace=True)
    df.index.name = "Pressure [dbar]"
    metadata = {"name": str(fname)}
    setattr(df, "_metadata", metadata)
    return df


def rosette_summary(fname):
    """
    Make a BTL (bottle) file from a ROS (bottle log) file.

    More control for the averaging process and at which step we want to
    perform this averaging eliminating the need to read the data into SBE
    Software again after pre-processing.
    NOTE: Do not run LoopEdit on the upcast!

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> fname = data_path.joinpath('CTD/g01l01s01.ros')
    >>> ros = ctd.rosette_summary(fname)
    >>> ros = ros.groupby(ros.index).mean()
    >>> ros.pressure.values.astype(int)
    array([835, 806, 705, 604, 503, 404, 303, 201, 151, 100,  51,   1])

    """
    ros = from_cnv(fname)
    ros["pressure"] = ros.index.values.astype(float)
    ros["nbf"] = ros["nbf"].astype(int)
    ros.set_index("nbf", drop=True, inplace=True, verify_integrity=False)
    return ros


def from_castaway_csv(fname):
    """
    DataFrame constructor to open CastAway CSV format.

    Example
    --------
    >>> import ctd
    >>> cast = ctd.from_castaway_csv('tests/data/castaway_data.csv')
    >>> downcast, upcast = cast.split() # Upcast often prefiltered
    >>> fig, ax = plt.subplots()
    >>> ax = downcast['temperature'].plot_cast()
    >>> fig.show()

    """
    with open(fname) as file:
        f = file.readlines()

    # Strip newline characters
    f = [s.strip() for s in f]

    # Separate meta data and CTD profile
    meta = [s for s in f if s[0] == "%"][0:-1]
    data = [s.split(",") for s in f if s[0] != "%"]
    df = pd.DataFrame(data[1:-1], columns=data[0])

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    # Normalise column names and extract units
    units = [s[s.find("(") + 1 : s.find(")")] for s in df.columns]
    df.columns = [
        _normalize_names(s.split("(")[0]).lower().replace(" ", "_") for s in df.columns
    ]
    df.set_index("pressure", drop=True, inplace=True, verify_integrity=False)

    # Add metadata
    meta = [s.replace("%", "").strip().split(",") for s in meta]
    metadata = {}
    for line in meta:
        metadata[line[0]] = line[1]
    metadata["units"] = units
    setattr(df, "_metadata", metadata)

    return df
