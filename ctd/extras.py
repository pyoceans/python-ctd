"""Extra functionality for plotting and post-processing."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ma


def _extrap1d(interpolator):
    """How to make scipy.interpolate return an extrapolated result beyond the
    input range.

    This is usually bad interpolation! But sometimes useful for pretty pictures,
    use it with caution!

    http://stackoverflow.com/questions/2745329/

    """
    xs, ys = interpolator.x, interpolator.y

    def pointwise(x):
        """Pointwise interpolation."""
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        if x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return interpolator(x)

    def ufunclike(xs):
        """Return an interpolation ufunc."""
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def get_maxdepth(self):
    """Return the maximum depth/pressure of a cast."""
    valid_last_depth = self.apply(pd.Series.notnull).to_numpy().T
    return np.float64(self.index.to_numpy() * valid_last_depth).max(axis=1)


def extrap_sec(
    data: np.ndarray,
    dist: np.ndarray,
    depth: np.ndarray,
    w1: float = 1.0,
    w2: float = 0,
) -> np.ndarray:
    """Extrapolate `data` to zones where the shallow stations are shadowed by
    the deep stations.  The shadow region usually cannot be extrapolates via
    linear interpolation.

    The extrapolation is applied using the gradients of the `data` at a certain
    level.

    Inputs
    ------
    data : Data to be extrapolated
    dist : Stations distance
    depth : Depth of the profile
    w1 : weights [0-1]
    w2 : weights [0-1]


    Outputs
    -------
    Sec_extrap : array_like
                 Extrapolated variable

    """
    from scipy.interpolate import interp1d

    new_data1 = []
    for row in data:
        new_row = row.copy()
        mask = ~np.isnan(row)
        if mask.any():
            y = row[mask]
            if y.size == 1:
                new_row = np.repeat(y, len(mask))
            else:
                x = dist[mask]
                f_i = interp1d(x, y)
                f_x = _extrap1d(f_i)
                new_row = f_x(dist)
        new_data1.append(new_row)

    new_data2 = []
    for col in data.T:
        new_col = col.copy()
        mask = ~np.isnan(col)
        if mask.any():
            y = col[mask]
            if y.size == 1:
                new_col = np.repeat(y, len(mask))
            else:
                z = depth[mask]
                f_i = interp1d(z, y)
                f_z = _extrap1d(f_i)
                new_col = f_z(depth)
        new_data2.append(new_col)

    return np.array(new_data1) * w1 + np.array(new_data2).T * w2


def gen_topomask(
    h: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dx: float = 1.0,
    kind: str = "linear",
) -> tuple:
    """Generate a topography mask from an oceanographic transect taking the
    deepest CTD scan as the depth of each station.

    Inputs
    ------
    h : array
        Pressure of the deepest CTD scan for each station [dbar].
    lons : array
           Longitude of each station [decimal degrees east].
    lat : Latitude of each station. [decimal degrees north].
    dx : float
         Horizontal resolution of the output arrays [km].
    kind : string, optional
           Type of the interpolation to be performed.
           See scipy.interpolate.interp1d documentation for details.

    Outputs
    -------
    xm : array
         Horizontal distances [km].
    hm : array
         Local depth [m].

    Author
    ------
    André Palóczy Filho (paloczy@gmail.com) --  October/2012

    """
    import gsw
    from scipy.interpolate import interp1d

    h, lon, lat = list(map(np.asanyarray, (h, lon, lat)))
    # Distance in km.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    h = -gsw.z_from_p(h, lat.mean())
    ih = interp1d(x, h, kind=kind, bounds_error=False, fill_value=h[-1])
    xm = np.arange(0, x.max() + dx, dx)
    hm = ih(xm)

    return xm, hm


def plot_section(  # noqa: PLR0915
    self: pd.DataFrame,
    *,
    reverse: bool = False,
    filled: bool = False,
    **kw: dict,
) -> tuple:
    """Plot a sequence of CTD casts as a section."""
    import gsw

    lon, lat, data = list(
        map(np.asanyarray, (self.lon, self.lat, self.to_numpy())),
    )
    data = ma.masked_invalid(data)
    h = self.get_maxdepth()
    if reverse:
        lon = lon[::-1]
        lat = lat[::-1]
        data = data.T[::-1].T
        h = h[::-1]
    lon, lat = map(np.atleast_2d, (lon, lat))
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    z = self.index.to_numpy().astype(float)

    if filled:  # CAVEAT: this method cause discontinuities.
        data = data.filled(fill_value=np.nan)
        data = extrap_sec(data, x, z, w1=0.97, w2=0.03)

    # Contour key words.
    extend = kw.pop("extend", "both")
    fontsize = kw.pop("fontsize", 12)
    labelsize = kw.pop("labelsize", 11)
    cmap = kw.pop("cmap", plt.cm.rainbow)
    levels = kw.pop(
        "levels",
        np.arange(np.floor(data.min()), np.ceil(data.max()) + 0.5, 0.5),
    )

    # Colorbar key words.
    pad = kw.pop("pad", 0.04)
    aspect = kw.pop("aspect", 40)
    shrink = kw.pop("shrink", 0.9)
    fraction = kw.pop("fraction", 0.05)

    # Topography mask key words.
    dx = kw.pop("dx", 1.0)
    kind = kw.pop("kind", "linear")
    linewidth = kw.pop("linewidth", 1.5)

    # Station symbols key words.
    station_marker = kw.pop("station_marker", None)
    color = kw.pop("color", "k")
    offset = kw.pop("offset", -5)
    alpha = kw.pop("alpha", 0.5)

    # Figure.
    figsize = kw.pop("figsize", (12, 6))
    fig, ax = plt.subplots(figsize=figsize)
    xm, hm = gen_topomask(h, lon, lat, dx=dx, kind=kind)
    ax.plot(xm, hm, color="black", linewidth=linewidth, zorder=3)
    ax.fill_between(xm, hm, y2=hm.max(), color="0.9", zorder=3)

    if station_marker:
        ax.plot(
            x,
            [offset] * len(h),
            color=color,
            marker=station_marker,
            alpha=alpha,
            zorder=5,
        )
    ax.set_xlabel("Cross-shore distance [km]", fontsize=fontsize)
    ax.set_ylabel("Depth [m]", fontsize=fontsize)
    ax.set_ylim(offset, hm.max())
    ax.invert_yaxis()

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")
    ax.xaxis.set_tick_params(tickdir="out", labelsize=labelsize, pad=1)
    ax.yaxis.set_tick_params(tickdir="out", labelsize=labelsize, pad=1)

    # Color version.
    cs = ax.contourf(
        x,
        z,
        data,
        cmap=cmap,
        levels=levels,
        alpha=1.0,
        extend=extend,
        zorder=2,
    )  # manual=True
    # Colorbar.
    cb = fig.colorbar(
        mappable=cs,
        ax=ax,
        orientation="vertical",
        aspect=aspect,
        shrink=shrink,
        fraction=fraction,
        pad=pad,
    )
    return fig, ax, cb


def cell_thermal_mass(
    temperature: pd.Series,
    conductivity: pd.Series,
) -> pd.Series:
    """Sample interval is measured in seconds.
    Temperature in degrees.
    CTM is calculated in S/m.

    """
    alpha = 0.03  # Thermal anomaly amplitude.
    beta = 1.0 / 7  # Thermal anomaly time constant (1/beta).

    sample_interval = 1 / 15.0
    a = 2 * alpha / (sample_interval * beta + 2)
    b = 1 - (2 * a / alpha)
    dc_o_dt = 0.1 * (1 + 0.006 * [temperature - 20])
    dt = np.diff(temperature)
    return -1.0 * b * conductivity + a * (dc_o_dt) * dt  # [S/m]


def mixed_layer_depth(ct: pd.Series, method: str = "half degree") -> pd.Series:
    """Return the mixed layer depth based on the "half degree" criteria."""
    half_degree = 0.5
    mask = (
        ct[0] - ct < half_degree
        if method == "half degree"
        else np.zeros_like(ct)
    )
    return pd.Series(mask, index=ct.index, name="MLD")


def barrier_layer_thickness(sa: pd.Series, ct: pd.Series) -> pd.Series:
    """Compute the thickness of water separating the mixed surface layer from
    the thermocline.
    A more precise definition would be the difference between mixed layer depth
    (MLD) calculated from temperature minus the mixed layer depth calculated
    using density.

    """
    import gsw

    sigma_theta = gsw.sigma0(sa, ct)
    mask = mixed_layer_depth(ct)
    mld = np.where(mask)[0][-1]
    sig_surface = sigma_theta[0]
    sig_bottom_mld = gsw.sigma0(sa[0], ct[mld])
    d_sig_t = sig_surface - sig_bottom_mld
    d_sig = sigma_theta - sig_bottom_mld
    mask = d_sig < d_sig_t  # Barrier layer.
    return pd.Series(mask, index=sa.index, name="BLT")
