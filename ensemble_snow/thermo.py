"""Small thermodynamic helpers (pure numpy)."""
from __future__ import annotations

import numpy as np

from . import config

RD_OVER_G = 287.04 / 9.80665                    # m K-1


def es_hpa(t_c):
    """Saturation vapour pressure over water (Bolton 1980), hPa."""
    t = np.asarray(t_c, np.float64)
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def td_from_rh_c(t_c, rh):
    """Dew point (degC) from temperature (degC) and relative humidity (percent)."""
    t = np.asarray(t_c, np.float64)
    g = np.log(np.clip(np.asarray(rh, np.float64), 1.0, 100.0) / 100.0) + 17.67 * t / (243.5 + t)
    return 243.5 * g / (17.67 - g)


def rh_from_td(t_c, td_c):
    return np.clip(100.0 * es_hpa(td_c) / es_hpa(t_c), 0.0, 100.0)


def wet_bulb_c(t_c, td_c, p_hpa):
    """Wet-bulb temperature (degC) by Newton iteration on the psychrometric equation (as in the reference viewer)."""
    t = np.asarray(t_c, np.float64)
    td = np.minimum(np.asarray(td_c, np.float64), t)
    p = np.asarray(p_hpa, np.float64)
    e = es_hpa(td)
    tw = 0.5 * (t + td)
    for _ in range(8):
        g = 6.62e-4 * (1.0 + 1.15e-3 * tw) * p
        f = es_hpa(tw) - g * (t - tw) - e
        df = es_hpa(tw) * 17.67 * 243.5 / (tw + 243.5) ** 2 + g
        tw = tw - f / df
    return np.minimum(tw, t).astype(np.float32)


def snow_fraction_thermo(tw_c, all_snow=None, all_rain=None):
    """Snow share of the precipitation from the 2 m wet-bulb temperature: 1 at/below ``all_snow``, 0 at/above ``all_rain``, linear between."""
    lo = config.TW_ALL_SNOW_C if all_snow is None else all_snow
    hi = config.TW_ALL_RAIN_C if all_rain is None else all_rain
    return np.clip((hi - np.asarray(tw_c, np.float64)) / (hi - lo), 0.0, 1.0).astype(np.float32)


def std_atmosphere_hpa(height_m):
    """Pressure (hPa) of the ICAO standard atmosphere at a height (m), good to ~0.5 hPa below 5 km."""
    h = np.asarray(height_m, np.float64)
    return 1013.25 * (1.0 - 2.25577e-5 * h) ** 5.25588


def std_atmosphere_height_m(p_hpa):
    return 44330.8 * (1.0 - (np.asarray(p_hpa, np.float64) / 1013.25) ** 0.190263)


def to_gpm(z, level_hpa):
    """Geopotential height in metres.  Some centres (MSC) publish decametres; decide by which reading sits closer to the standard atmosphere."""
    z = np.asarray(z, np.float32)
    med = float(np.nanmedian(z))
    expect = float(std_atmosphere_height_m(level_hpa))
    return z * 10.0 if abs(med * 10.0 - expect) < abs(med - expect) else z


def layer_thickness_m(levels, t_k, p_bot, p_top):
    """Hypsometric thickness (m) between two of ``levels`` from the layer temperatures (dry-air approximation: a few metres low)."""
    order = np.argsort([-p for p in levels])
    p = [float(levels[i]) for i in order]
    idx = [k for k, x in enumerate(p) if p_top <= x <= p_bot]
    dz = np.zeros(np.shape(t_k)[1:], np.float64)
    for i, j in zip(idx[:-1], idx[1:]):
        dz += 0.5 * (np.asarray(t_k[order[i]], np.float64) + t_k[order[j]]) * np.log(p[i] / p[j])
    return (RD_OVER_G * dz).astype(np.float32)
