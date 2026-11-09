"""Snow-to-liquid ratio (SLR) methods: a registry of pure functions, one decorator each.

How to add a method
-------------------
1. Write a function ``fn(col: Column) -> ndarray`` that returns the ratio (inches of snow per inch of liquid) with the same shape as
   ``col.shape`` (``(members, ny, nx)``).  Return NaN where the method cannot be evaluated.
2. Decorate it with :func:`register`, naming the ``Column`` inputs it needs.  A system that does not publish an input simply does not
   get the method (the manifest records which methods each system has).
3. Optionally list its key in ``config.BLEND`` to include it in the multi-method consensus.
4. Add a line to ``tests/test_slr.py``.  The page, the statistics and the files pick the new method up from the manifest: no
   front-end change is needed.

Methods that depend on the precipitation of the interval itself (``interval=True``, e.g. Byun) receive ``col.precip_mm3h``; all other
methods are evaluated at both ends of a forecast interval and averaged.

Sources (the formulas are generalised from the RRFS winter viewer's ``winter.py``; coefficients are unchanged)
* Kuchera & Veals: Kuchera (2004), see Veals et al. (2025) eq. 2.
* Max-T-aloft: Veals et al. (2025) eq. 1 (fitted to Alaska data, so experimental here).
* CMC thickness ("Murphy factor") and NBM thickness: CMC operational snow-depth guidance; NBM v5.0 winter-suite documentation.
* Byun et al. (2008), Weather and Forecasting 23, 644-658.
* Baxter, Graves & Moore (2005), Weather and Forecasting 20, 729-744, Table 2 (climatology).
* Cobb & Waldstreicher (2005), as implemented in the National Blend of Models (2011 temperature curve, humidity-weighted layers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config, thermo
from .fields import Profile

SLR_MIN, SLR_MAX = 3.0, 40.0          # ratios outside 3:1..40:1 are not meaningful for a model-column statistic
KUCHERA_PIVOT_K = 271.16


# --------------------------------------------------------------------------- the input bundle
@dataclass
class Column:
    """The model column at one time, on the display grid (``(M, ny, nx)`` surface fields, :class:`Profile` upper-air fields)."""
    shape: tuple[int, ...]
    lat: np.ndarray                                   # (ny, nx)
    lon: np.ndarray
    t2_k: np.ndarray | None = None
    ps_hpa: np.ndarray | None = None                  # surface pressure; None -> every level counts as above ground
    t: Profile | None = None
    rh: Profile | None = None
    z: Profile | None = None
    precip_mm3h: np.ndarray | None = None             # liquid precipitation of the interval scaled to mm per 3 h (interval methods only)

    def ground_hpa(self) -> np.ndarray:
        return self.ps_hpa if self.ps_hpa is not None else np.full(self.shape, 1100.0, np.float32)


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class SlrMethod:
    key: str
    label: str
    fn: Callable[[Column], np.ndarray]
    needs: tuple[str, ...]
    desc: str
    ref: str
    group: str = "method"          # "method" (a forecasting technique) | "reference" (a fixed ratio or a climatology)
    interval: bool = False

    @property
    def in_blend(self) -> bool:
        return self.key in config.BLEND

    def available(self, col: Column) -> bool:
        return all(getattr(col, n, None) is not None for n in self.needs)


METHODS: dict[str, SlrMethod] = {}


def register(key: str, label: str, needs: tuple[str, ...] = (), *, desc: str, ref: str, group: str = "method", interval: bool = False):
    def deco(fn):
        if key in METHODS:
            raise ValueError(f"duplicate SLR method {key!r}")
        METHODS[key] = SlrMethod(key, label, fn, tuple(needs), desc, ref, group, interval)
        return fn
    return deco


def interval_ratio(m: SlrMethod, col_prev: Column | None, col_end: Column) -> np.ndarray:
    """The ratio applied to the precipitation of one forecast interval."""
    r1 = m.fn(col_end)
    if m.interval or col_prev is None or not m.available(col_prev):
        return r1
    r0 = m.fn(col_prev)
    return np.where(np.isnan(r0), r1, np.where(np.isnan(r1), r0, 0.5 * (r0 + r1))).astype(np.float32)


def _clip(a):
    return np.clip(a, SLR_MIN, SLR_MAX).astype(np.float32)


def _lev(levels, ndim):
    return np.asarray(levels, np.float32).reshape((-1,) + (1,) * (ndim - 1))


# --------------------------------------------------------------------------- column helpers
def tmax_column_k(t2_k, prof: Profile, ps_hpa, p_top=500):
    """Warmest temperature (K) from the surface (2 m) to ``p_top`` hPa; levels below ground are ignored."""
    p = _lev(prof.levels, prof.data.ndim)
    ok = (p <= ps_hpa[None]) & (p >= p_top)
    t = np.where(ok, prof.data, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmax(np.concatenate([t, t2_k[None]], axis=0), axis=0)


def tmax_aloft_c(t2_k, prof: Profile, ps_hpa, p_top=500, agl_hpa=70.0):
    """Max temperature (degC) from ~610 m AGL (ps - ``agl_hpa``) to ``p_top``; falls back to 2 m where the layer is empty."""
    p = _lev(prof.levels, prof.data.ndim)
    ok = (p <= ps_hpa[None] - agl_hpa) & (p >= p_top)
    t = np.where(ok, prof.data, np.nan)
    with np.errstate(all="ignore"):
        m = np.nanmax(t, axis=0)
    return (np.where(np.isfinite(m), m, t2_k) - 273.15).astype(np.float32)


def _thickness(col: Column, p_bot: int, p_top: int) -> np.ndarray:
    """Layer thickness (m): model heights when the column has both levels, else the hypsometric equation on the temperature profile."""
    if col.z is not None and p_bot in col.z.levels and p_top in col.z.levels:
        return (col.z.at(p_top) - col.z.at(p_bot)).astype(np.float32)
    return thermo.layer_thickness_m(col.t.levels, col.t.data, p_bot, p_top)


# --------------------------------------------------------------------------- fixed ratios
@register("fixed10", "10:1", desc="Fixed 10:1 ratio: the textbook rule of thumb.", ref="Rule of thumb", group="reference")
def slr_fixed10(col):
    return np.full(col.shape, 10.0, np.float32)


@register("fixed12", "12:1", desc="Fixed 12:1 ratio, closer to New England climatology than 10:1.", ref="Baxter et al. (2005)", group="reference")
def slr_fixed12(col):
    return np.full(col.shape, 12.0, np.float32)


# --------------------------------------------------------------------------- temperature-based methods
@register("kuchera", "Kuchera", ("t2_k", "t"),
          desc="Starts at 12:1 from the warmest temperature between the surface and 500 hPa: gains 1 point per degree below -2 °C (271.16 K) and loses 2 points per degree above it.",
          ref="Kuchera (2004); Veals et al. (2025) eq. 2")
def slr_kuchera(col):
    tmax = tmax_column_k(col.t2_k, col.t, col.ground_hpa(), 500)
    return _clip(np.where(tmax > KUCHERA_PIVOT_K, 12.0 + 2.0 * (KUCHERA_PIVOT_K - tmax), 12.0 + (KUCHERA_PIVOT_K - tmax)))


_MAXT_COEF = (4.5e-6, 4.432e-4, 0.0130903, 0.0585968, -1.8150809, 5.9805722)      # T^5 .. T^0


@register("maxt", "Max-T-aloft", ("t2_k", "t"),
          desc="5th-degree polynomial in the warmest temperature between ~610 m above ground and 500 hPa. Experimental: fitted to Alaska data.",
          ref="Veals et al. (2025) eq. 1")
def slr_maxt(col):
    t = np.clip(tmax_aloft_c(col.t2_k, col.t, col.ground_hpa()), -35.0, 5.0)        # keep the polynomial inside its fitted range
    return _clip(np.polyval(_MAXT_COEF, t.astype(np.float64)))


# --------------------------------------------------------------------------- thickness methods
CMC_CRITICAL_850_700_M = 1540.0


@register("cmc_thick", "CMC thickness", ("t",),
          desc="10:1, reduced by the Murphy relation 10 (30.6 - 0.1057 Z1000-700[dam]) where the 850-700 hPa thickness exceeds 1540 m (marginal, wet snow).",
          ref="CMC operational snow-depth guidance (Murphy factor)")
def slr_cmc_thick(col):
    dz_1000_700 = _thickness(col, 1000, 700).astype(np.float64)
    dz_850_700 = _thickness(col, 850, 700)
    murphy = 10.0 * (-0.1057 * (dz_1000_700 / 10.0) + 30.6)
    return _clip(np.where(dz_850_700 > CMC_CRITICAL_850_700_M, np.minimum(murphy, 10.0), 10.0))


@register("nbm_thick", "NBM thickness", ("t",),
          desc="A straight line in the 850-700 hPa thickness: -0.16559 Z[m] + 263.35 (8:1 at 1540 m, 15:1 at 1500 m).",
          ref="NBM v5.0 winter-suite documentation")
def slr_nbm_thick(col):
    return _clip(-0.16559 * _thickness(col, 850, 700).astype(np.float64) + 263.35)


# --------------------------------------------------------------------------- Byun (logistic, depends on the interval's precipitation)
# Byun et al. 2008: (upper bound of 3-h precipitation [mm], a, b, c)
_BYUN = ((2.0, 18.8, 0.0811, 0.6508), (3.0, 16.1, 0.2182, 0.5373), (4.0, 14.9, 0.2295, 0.5174),
         (5.0, 13.2, 0.2678, 0.5074), (np.inf, 11.9, 0.1524, 0.5174))


@register("byun", "Byun", ("t2_k", "precip_mm3h"), interval=True,
          desc="Logistic function of the 2 m temperature with coefficients chosen by the 3-hour precipitation. Fitted to Korean data: it never exceeds ~19:1, "
               "so it is a wet-snow-sensitive lower bound rather than a cold-powder estimate.",
          ref="Byun et al. (2008)")
def slr_byun(col):
    t = col.t2_k.astype(np.float64) - 273.15
    pr = np.asarray(col.precip_mm3h, np.float64)
    a = np.full(t.shape, _BYUN[-1][1])
    b = np.full(t.shape, _BYUN[-1][2])
    c = np.full(t.shape, _BYUN[-1][3])
    for hi, ca, cb, cc in reversed(_BYUN[:-1]):
        m = pr < hi
        a, b, c = np.where(m, ca, a), np.where(m, cb, b), np.where(m, cc, c)
    with np.errstate(over="ignore"):
        return (a / (1.0 + np.exp(np.clip((t - b) / c, -60.0, 60.0)))).astype(np.float32)


# --------------------------------------------------------------------------- Baxter climatology
# Baxter et al. 2005, Table 2: mean SLR of the NWS county warning areas in and around New England (lat, lon of the office, mean).
BAXTER_CWA = (
    ("Caribou, ME", 46.87, -68.02, 12.3), ("Burlington, VT", 44.47, -73.15, 13.4), ("Gray/Portland, ME", 43.89, -70.25, 11.6),
    ("Albany, NY", 42.69, -73.83, 12.2), ("Binghamton, NY", 42.21, -75.98, 13.0), ("Boston/Taunton, MA", 41.91, -71.07, 10.7),
    ("Upton, NY", 40.87, -72.86, 10.6),
)


def baxter_field(lat, lon, power: float = 2.0, floor_km: float = 60.0):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    num, den = np.zeros(la.shape), np.zeros(la.shape)
    for _, slat, slon, val in BAXTER_CWA:
        dx = (lo - np.radians(slon)) * np.cos(0.5 * (la + np.radians(slat)))
        dy = la - np.radians(slat)
        d = np.maximum(6371.0 * np.hypot(dx, dy), floor_km)
        w = d ** -power
        num += w * val
        den += w
    return (num / den).astype(np.float32)


@register("climo", "Climatological", desc="The regional mean ratio of the nearest NWS offices (inverse-distance weighted, 10.6-13.4:1). A reference that does not respond to the forecast.",
          ref="Baxter, Graves & Moore (2005), Table 2", group="reference")
def slr_climo(col):
    return np.broadcast_to(baxter_field(col.lat, col.lon), col.shape).astype(np.float32).copy()


# --------------------------------------------------------------------------- Cobb layer ratio
COBB_T_C = np.array([-300, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 1, 300], float)
COBB_SR = np.array([7.2, 7.2, 6.8, 7.0, 8.8, 12.0, 18.0, 23.0, 26.0, 22.5, 17.5, 12.0, 9.5, 9.0, 8.5, 7.0, 3.0, 0.0, 0.0], float)
COBB_P_BOT, COBB_P_STEP = 925, 25
COBB_MAX = 30.0


def cobb_layer_sr(t_c):
    return np.interp(t_c, COBB_T_C, COBB_SR)


def _lnp_interp(pt: float, lev: np.ndarray, arr: np.ndarray):
    """Linear-in-ln(p) interpolation to ``pt``.  ``lev`` descending (hPa); NaN (underground) neighbours are skipped; None outside the data."""
    if pt > lev[0] + 1e-6 or pt < lev[-1] - 1e-6:
        return None
    i = int(np.searchsorted(-lev, -pt, side="left"))
    lo, hi = (i, i) if abs(lev[i] - pt) < 1e-6 else (i - 1, i)
    w = 0.0 if lo == hi else np.log(pt / lev[lo]) / np.log(lev[hi] / lev[lo])
    a_lo, a_hi = arr[lo], arr[hi]
    return np.where(np.isnan(a_lo), a_hi, np.where(np.isnan(a_hi), a_lo, a_lo + w * (a_hi - a_lo)))


@register("cobb", "Cobb", ("t", "rh"),
          desc="Cloud-layer ratio: the 2011 Cobb temperature curve (peak 26:1 near -16 C) averaged over every 25 hPa layer from 925 hPa to the top level, "
               "weighted by (RH/80)^2, capped at 30:1. The vertical-velocity weight NBM uses is not applied (humidity-weighted layer mean).",
          ref="Cobb & Waldstreicher (2005); NBM 2011 curve")
def slr_cobb(col, top_hpa: float | None = None):
    top = config.COBB_TOP_HPA if top_hpa is None else top_hpa
    ps = col.ground_hpa()
    tp, rp = col.t.sorted_desc(), col.rh.sorted_desc()

    def masked(prof):
        lev = np.asarray(prof.levels, np.float64)
        ok = lev.reshape((-1,) + (1,) * (prof.data.ndim - 1)) <= ps[None] + 1e-6
        return lev, np.where(ok, prof.data, np.nan).astype(np.float64)

    (tl, t_c), (rl, rh) = masked(tp), masked(rp)
    t_c = t_c - 273.15
    num, den = np.zeros(col.shape), np.zeros(col.shape)
    pt = COBB_P_BOT
    while pt >= top - 1e-6:
        t, r = _lnp_interp(pt, tl, t_c), _lnp_interp(pt, rl, rh)
        if t is not None and r is not None:
            good = np.isfinite(t) & np.isfinite(r) & (pt <= ps + 1e-6)
            wt = np.where(good, np.clip(r, 0.0, 100.0) ** 2 / 6400.0, 0.0)
            num += np.where(good, cobb_layer_sr(np.where(good, t, 0.0)), 0.0) * wt
            den += wt
        pt -= COBB_P_STEP
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.minimum(out, COBB_MAX).astype(np.float32)
