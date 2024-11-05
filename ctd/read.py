"""Read module."""

from __future__ import annotations

import bz2
import collections
import datetime
import gzip
import linecache
import re
import warnings
import zipfile
from io import StringIO
from pathlib import Path

import chardet
import gsw
import numpy as np
import pandas as pd


def _basename(fname: str | Path) -> (str, str, str):
    """Return file name without path."""
    if not isinstance(fname, Path):
        fname = Path(fname)
    path, name, ext = fname.parent, fname.stem, fname.suffix
    return path, name, ext


def _normalize_names(name: str) -> str:
    """Normalize column names."""
    name = name.strip()
    return name.strip("*")


def _open_compressed(fname: Path) -> str:
    """Open compressed gzip, gz, zip or bz2 files."""
    extension = fname.suffix.casefold()
    loaders = {
        ".gzip": gzip.open,
        ".gz": gzip.open,
        ".bz2": bz2.BZ2File,
        ".zip": zipfile.ZipFile,
    }
    loader = loaders.get(extension)
    if loader is None:
        valid = ", ".join(loaders.keys())
        msg = (
            "Unrecognized file extension. "
            f"Expected {valid}, got {extension}."
        )
        raise ValueError(msg)

    if extension == ".zip":
        # NOTE: Zip format may contain more than one file in the archive
        # (similar to tar), here we assume that there is just one file per
        # zipfile!  Also, we ask for the name because it can be different from
        # the zipfile file!!
        with loader(str(fname)) as zfile:
            name = zfile.namelist()[0]
            with zfile.open(name) as cfile:
                return cfile.read()
    with loader(str(fname)) as cfile:
        return cfile.read()


def _read_file(fname: str | Path | StringIO) -> StringIO:
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
    elif extension in [".cnv", ".edf", ".txt", ".ros", ".btl", ".bl", ".csv"]:
        contents = fname.read_bytes()
    else:
        msg = (
            "Unrecognized file extension. "
            f"Expected .cnv, .edf, .txt, .ros, or .btl got {extension}"
        )
        raise ValueError(
            msg,
        )
    # Read as bytes but we need to return strings for the parsers.
    encoding = chardet.detect(contents)["encoding"]
    text = contents.decode(encoding=encoding, errors="replace")
    return StringIO(text)


def _remane_duplicate_columns(names: str) -> str:
    """Rename a column when it is duplicated."""
    items = collections.Counter(names).items()
    dup = []
    for item, count in items:
        if count > 2:  # noqa: PLR2004
            msg = (
                "Cannot handle more than two duplicated columns. "
                f"Found {count} for {item}."
            )
            raise ValueError(
                msg,
            )
        if count > 1:
            dup.append(item)

    # We can assume there are only two instances of a word in the list,
    # we find the last index of an instance,
    # which will be the second occurrence of the item.
    second_occurrences = [
        len(names) - names[::-1].index(item) - 1 for item in dup
    ]
    for idx in second_occurrences:
        names[idx] = f"{names[idx]}_"
    return names


def _parse_seabird(lines: list, ftype: str) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Parse searbird formats."""
    # Initialize variables.
    lon = lat = time = None, None, None
    fname = None
    skiprows = 0

    metadata = {}
    header, config, names = [], [], []
    for k, raw_line in enumerate(lines):
        line = raw_line.strip()

        # Only cnv has columns names,
        # for bottle files we will use the variable row.
        if ftype == "cnv" and "# name" in line:
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
            lat = np.float64(lat.split())
            if hemisphere == "S":
                lat = -(lat[0] + lat[1] / 60.0)
            elif hemisphere == "N":
                lat = lat[0] + lat[1] / 60.0
            else:
                msg = "Latitude not recognized."
                raise ValueError(msg)
        if "NMEA Longitude" in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split("=")[1].strip()
            lon = np.float64(lon.split())
            if hemisphere == "W":
                lon = -(lon[0] + lon[1] / 60.0)
            elif hemisphere == "E":
                lon = lon[0] + lon[1] / 60.0
            else:
                msg = "Latitude not recognized."
                raise ValueError(msg)
        if "NMEA UTC (Time)" in line:
            time = line.split("=")[-1].strip()
            # Should use some fuzzy datetime parser to make this more robust.
            time = datetime.datetime.strptime(
                time,
                "%b %d %Y %H:%M:%S",
            ).astimezone(datetime.UTC)

        # cnv file header ends with *END* while
        if ftype == "cnv":
            if line == "*END*":
                skiprows = k + 1
                break
        else:  # btl.
            # There is no *END* like in a .cnv file, skip two after header info.
            # Skip empty lines.
            if not line:
                continue

            if not (line.startswith("*") | line.startswith("#")):
                # Fix commonly occurring problem when Sbeox.* exists in the file
                # the name is concatenated to previous parameter
                # example:
                #   CStarAt0Sbeox0Mm/Kg to CStarAt0 Sbeox0Mm/Kg
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


def from_bl(fname: str | Path) -> pd.DataFrame:
    """Read Seabird bottle-trip (bl) file.

    Example:
    -------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> df = ctd.from_bl(str(data_path.joinpath("bl", "bottletest.bl")))
    >>> df._metadata["time_of_reset"]
    datetime.datetime(2018, 6, 25, 20, 8, 55)

    """
    f = _read_file(fname)
    cast = pd.read_csv(
        f,
        skiprows=2,
        parse_dates=[1],
        index_col=0,
        names=["bottle_number", "time", "startscan", "endscan"],
    )
    cast._metadata = {  # noqa: SLF001
        "time_of_reset": pd.to_datetime(
            linecache.getline(str(fname), 2)[6:-1],
        ).to_pydatetime(),
    }
    return cast


def from_btl(fname: str | Path) -> pd.DataFrame:
    """DataFrame constructor to open Seabird CTD BTL-ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> bottles = ctd.from_btl(data_path.joinpath("btl", "bottletest.btl"))

    """
    f = _read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype="btl")

    f.seek(0)

    cast = pd.read_fwf(
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
    rowtypes = cast[cast.columns[-1]].unique()
    # Get times and dates which occur on second line of each bottle.
    date_idx = metadata["names"].index("Date")
    dates = cast.iloc[:: len(rowtypes), date_idx].reset_index(drop=True)
    times = cast.iloc[1 :: len(rowtypes), date_idx].reset_index(drop=True)
    datetimes = dates + " " + times

    # Fill the Date column with datetimes.
    cast.loc[:: len(rowtypes), "Date"] = datetimes.to_numpy()
    cast.loc[1 :: len(rowtypes), "Date"] = datetimes.to_numpy()

    # Fill missing rows.
    cast["Bottle"] = cast["Bottle"].ffill()
    cast["Date"] = cast["Date"].ffill()

    cast["Statistic"] = (
        cast["Statistic"].str.lstrip("(").str.rstrip(")")
    )  # (avg) to avg

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
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = cast[column].astype(float)
            except ValueError:
                warnings.warn(
                    f"Could not convert {column} to float.",
                    stacklevel=2,
                )

    cast["Date"] = pd.to_datetime(cast["Date"])
    cast._metadata = metadata  # noqa: SLF001
    return cast


def from_edf(fname: str | Path) -> pd.DataFrame:  # noqa: C901, PLR0912
    """DataFrame constructor to open XBT EDF ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_edf(data_path.joinpath("XBT.EDF.gz"))
    >>> ax = cast["temperature"].plot_cast()

    """
    f = _read_file(fname)
    header, names = [], []
    for k, raw_line in enumerate(f.readlines()):
        line = raw_line.strip()
        if line.startswith("Serial Number"):
            serial = line.strip().split(":")[1].strip()
        elif line.startswith("Latitude"):
            try:
                hemisphere = line[-1]
                lat = line.strip(hemisphere).split(":")[1].strip()
                lat = np.float64(lat.split())
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
                lon = np.float64(lon.split())
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
    cast = pd.read_csv(
        f,
        header=None,
        index_col=None,
        names=names,
        skiprows=skiprows,
        sep=r"\s+",
    )
    f.close()

    cast = cast.set_index("depth", drop=True)
    cast.index.name = "Depth [m]"
    name = _basename(fname)[1]

    metadata = {
        "lon": lon,
        "lat": lat,
        "name": str(name),
        "header": "\n".join(header),
        "serial": serial,
    }
    cast._metadata = metadata  # noqa: SLF001
    return cast


def from_cnv(fname: str | Path) -> pd.DataFrame:
    """DataFrame constructor to open Seabird CTD CNV-ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_cnv(data_path.joinpath("CTD_big.cnv.bz2"))
    >>> downcast, upcast = cast.split()
    >>> ax = downcast["t090C"].plot_cast()

    """
    f = _read_file(fname)
    metadata = _parse_seabird(f.readlines(), ftype="cnv")

    f.seek(0)
    cast = pd.read_fwf(
        f,
        header=None,
        index_col=None,
        names=metadata["names"],
        skiprows=metadata["skiprows"],
        sep=r"\s+",
        widths=[11] * len(metadata["names"]),
    )
    f.close()

    prkeys = [
        "prM",
        "prE",
        "prDM",
        "pr50M",
        "pr50M1",
        "prSM",
        "prdM",
        "pr",
        "depSM",
        "prDE",
    ]
    cast.columns = cast.columns.str.strip()
    prkey = [key for key in prkeys if key in cast.columns]
    if len(prkey) == 0:
        msg = "Expected one pressure/depth column, didn't receive any"
        raise ValueError(
            msg,
        )
    if len(prkey) > 1:
        # If multiple keys present then keep the first one.
        prkey = prkey[0]

    cast = cast.set_index(prkey, drop=True)
    cast.index.name = "Pressure [dbar]"
    if prkey == "depSM":
        lat = metadata.get("lat", None)
        if lat is not None:
            cast.index = gsw.p_from_z(
                cast.index,
                lat,
                geo_strf_dyn_height=0,
                sea_surface_geopotential=0,
            )
        else:
            msg = (
                "Missing latitude information. Cannot compute pressure! "
                f"Your index is {prkey}, please compute pressure manually "
                "with `gsw.p_from_z` and overwrite your index."
            )
            warnings.war(msg)
            cast.index.name = prkey

    if "name" not in metadata:
        name = _basename(fname)[1]
        metadata["name"] = str(name)

    dtypes = {"bpos": int, "pumps": bool, "flag": bool}
    for column in cast.columns:
        if column in dtypes:
            cast[column] = cast[column].astype(dtypes[column])
        else:
            try:
                cast[column] = cast[column].astype(float)
            except ValueError:
                warnings.warn(
                    f"Could not convert {column} to float.",
                    stacklevel=2,
                )

    cast._metadata = metadata  # noqa: SLF001
    return cast


def from_fsi(fname: str | Path, skiprows: int = 9) -> pd.DataFrame:
    """DataFrame constructor to open Falmouth Scientific, Inc. (FSI) CTD
    ASCII format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> cast = ctd.from_fsi(data_path.joinpath("FSI.txt.gz"))
    >>> downcast, upcast = cast.split()
    >>> ax = downcast["TEMP"].plot_cast()

    """
    f = _read_file(fname)
    fsi = pd.read_csv(
        f,
        header="infer",
        index_col=None,
        skiprows=skiprows,
        dtype=float,
        sep=r"\s+",
    )
    f.close()

    fsi = fsi.set_index("PRES", drop=True)
    fsi.index.name = "Pressure [dbar]"
    metadata = {"name": str(fname)}
    fsi._metadata = metadata  # noqa: SLF001
    return fsi


def rosette_summary(fname: str | Path) -> pd.DataFrame:
    """Make a BTL (bottle) file from a ROS (bottle log) file.

    More control for the averaging process and at which step we want to
    perform this averaging eliminating the need to read the data into SBE
    Software again after pre-processing.
    NOTE: Do not run LoopEdit on the upcast!

    Examples
    --------
    >>> from pathlib import Path
    >>> import ctd
    >>> data_path = Path(__file__).parents[1].joinpath("tests", "data")
    >>> fname = data_path.joinpath("CTD/g01l01s01.ros")
    >>> ros = ctd.rosette_summary(fname)
    >>> ros = ros.groupby(ros.index).mean()
    >>> ros.pressure.to_numpy().astype(int)
    array([835, 806, 705, 604, 503, 404, 303, 201, 151, 100,  51,   1])

    """
    ros = from_cnv(fname)
    ros["pressure"] = ros.index.to_numpy().astype(float)
    ros["nbf"] = ros["nbf"].astype(int)
    metadata = ros._metadata  # noqa: SLF001
    ros = ros.set_index("nbf", drop=True, verify_integrity=False)
    ros._metadata = metadata  # noqa: SLF001
    return ros


def from_castaway_csv(fname: str | Path) -> pd.DataFrame:
    """DataFrame constructor to open CastAway CSV format.

    Example:
    -------
    >>> import ctd
    >>> cast = ctd.from_castaway_csv("tests/data/castaway_data.csv")
    >>> cast.columns
    Index(['depth', 'temperature', 'conductivity', 'specific_conductance',
           'salinity', 'sound_velocity', 'density'],
          dtype='object')

    """
    f = _read_file(fname)
    lines = f.readlines()

    # Strip newline characters
    lines = [s.strip() for s in lines]

    # Separate meta data and CTD profile
    meta = [s for s in lines if s[0] == "%"][0:-1]
    data = [s.split(",") for s in lines if s[0] != "%"]
    cast = pd.DataFrame(data[1:-1], columns=data[0])

    # Convert to numeric
    for col in cast.columns:
        cast[col] = pd.to_numeric(cast[col])

    # Normalise column names and extract units
    units = [s[s.find("(") + 1 : s.find(")")] for s in cast.columns]
    cast.columns = [
        _normalize_names(s.split("(")[0]).lower().replace(" ", "_")
        for s in cast.columns
    ]
    cast = cast.set_index("pressure", drop=True, verify_integrity=False)

    # Add metadata
    meta = [s.replace("%", "").strip().split(",") for s in meta]
    metadata = {}
    for line in meta:
        metadata[line[0]] = line[1]
    metadata["units"] = units
    cast._metadata = metadata  # noqa: SLF001

    return cast
