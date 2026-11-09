"""ECCC / CMC GEPS (21 members) from the MSC Datamart.

One GRIB2 file per variable, level and forecast hour holds all members ("allmbrs"):

    https://dd.weather.gc.ca/today/ensemble/geps/grib2/raw/HH/hhh/CMC_geps-raw_<VAR>_<LEVELTYPE>_<LEVEL>_latlon0p5x0p5_YYYYMMDDHH_Phhh_allmbrs.grib2

Names used here were read from a live directory listing of the Datamart (APCP_SFC_0, ASNOW_SFC_0, TMP_TGL_2m, RH_TGL_2m, TMP/RH/HGT_ISBL_<hPa>); 600 hPa is not
published for GEPS and is skipped.  ``PRES_SFC_0`` (surface pressure) is requested but optional.  The Datamart keeps only a short rolling window of
runs, so a run that was missed cannot be recovered later.

ASNOW's units are not stated in the MSC documentation we could reach.  It is used as the model's own snow-type liquid only if it passes a plausibility
check (never larger than the total precipitation); otherwise the wet-bulb ramp is used and the manifest says so.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from .. import config, thermo
from ..accum import CumulativeTracker
from ..fields import EnsembleData
from ..grib import DECODE_LOCK, Msg, iter_messages
from ..http import Fetcher, NotFound
from . import common

log = logging.getLogger(__name__)
BASE = "https://dd.weather.gc.ca/today/ensemble/geps/grib2/raw"


def url(cycle: datetime, var: str, ltype: str, level: str, fhr: int) -> str:
    stamp = f"{cycle:%Y%m%d%H}"
    return f"{BASE}/{cycle:%H}/{fhr:03d}/CMC_geps-raw_{var}_{ltype}_{level}_latlon0p5x0p5_{stamp}_P{fhr:03d}_allmbrs.grib2"


def snow_is_water_equivalent(snow_cum: np.ndarray, precip_cum: np.ndarray) -> bool:
    """ASNOW looks like liquid-equivalent snow if it never exceeds the total precipitation by more than rounding (a depth would be ~10x larger)."""
    ok = precip_cum >= 2.0
    if ok.sum() < 5:
        return True
    ratio = snow_cum[ok] / precip_cum[ok]
    return bool(np.nanpercentile(ratio, 99) <= 1.2)


class GepsSource:
    key = "geps"

    def __init__(self, fetcher: Fetcher | None = None):
        self.http = fetcher or Fetcher()

    def is_complete(self, cycle: datetime, max_fhr: int) -> bool:
        return self.http.exists(url(cycle, "APCP", "SFC", "0", max_fhr))

    def _fetch(self, cycle, var, ltype, level, fhr) -> list[Msg] | None:
        try:
            data = self.http.get(url(cycle, var, ltype, level, fhr))
        except NotFound:
            return None
        with DECODE_LOCK:
            return list(iter_messages(data))

    def _task(self, cycle, fhr):
        want = [("APCP", "SFC", "0"), ("ASNOW", "SFC", "0"), ("TMP", "TGL", "2m"), ("RH", "TGL", "2m"), ("PRES", "SFC", "0")]
        want += [("TMP", "ISBL", f"{p:04d}") for p in config.T_LEVELS]
        want += [("RH", "ISBL", f"{p:04d}") for p in config.RH_LEVELS]
        want += [("HGT", "ISBL", f"{p:04d}") for p in config.Z_LEVELS]
        out = {}
        for var, lt, lev in want:
            if fhr == 0 and var in ("APCP", "ASNOW"):
                continue
            out[(var, lt, lev)] = self._fetch(cycle, var, lt, lev, fhr)
        return fhr, out

    def load(self, cycle: datetime, steps: list[int], workers: int = 8) -> EnsembleData:
        data: dict[int, dict] = {}
        with ThreadPoolExecutor(workers) as ex:
            for fhr, out in ex.map(lambda f: self._task(cycle, f), steps):
                data[fhr] = out

        def msgs(f, var, lt, lev):
            return data.get(f, {}).get((var, lt, lev))

        first = next((msgs(f, "APCP", "SFC", "0") for f in steps if f > 0 and msgs(f, "APCP", "SFC", "0")), None)
        if not first:
            raise RuntimeError("GEPS: no precipitation found (check the cycle and the Datamart path)")
        members = sorted(common.member_ids(first))
        notes: list[str] = []

        def cumulative(var):
            """Total since initialisation per step on the display grid, resolved from the step ranges."""
            tr, axes = CumulativeTracker(), None
            for f in steps:
                ms = msgs(f, var, "SFC", "0")
                if f > 0 and ms:
                    arr, lat, lon = common.stack(ms, members, common.water_scale(ms[0]))
                    a, b = ms[0].step_range
                    tr.add(a if b > a else 0, b if b > a else f, arr)
                    axes = (lat, lon)
            if axes is None:
                return None
            rg = common.regridder_for(*axes)
            return {f: rg(v) for f, v in tr.cum.items() if f > 0 and f in steps and not isinstance(v, float)}

        precip = cumulative("APCP")
        snow = cumulative("ASNOW")
        use_snow = False
        if snow:
            last = max(set(snow) & set(precip))
            use_snow = snow_is_water_equivalent(snow[last], precip[last])
            if not use_snow:
                notes.append("GEPS ASNOW does not look like liquid-equivalent snow; precipitation type estimated from the 2 m wet-bulb temperature instead")

        def disp(var, lt, lev, f, scale=1.0, offset=0.0, fix=None):
            ms = msgs(f, var, lt, lev)
            if not ms:
                return None
            if fix is not None:
                for m in ms:
                    m.values = fix(m.values)
            arr, la, lo = common.stack(ms, members, scale, offset)
            return common.regridder_for(la, lo)(arr)

        out_steps = []
        for f in steps:
            kw: dict = {}
            if f > 0:
                kw["precip_cum_mm"] = precip.get(f)
                if use_snow:
                    kw["snow_liq_cum_mm"] = snow.get(f)
            kw["t2m_k"] = disp("TMP", "TGL", "2m", f)
            kw["rh2m"] = disp("RH", "TGL", "2m", f)
            kw["ps_hpa"] = disp("PRES", "SFC", "0", f, 0.01)
            kw["t"] = common.profile({p: a for p in config.T_LEVELS if (a := disp("TMP", "ISBL", f"{p:04d}", f)) is not None})
            kw["rh"] = common.profile({p: a for p in config.RH_LEVELS if (a := disp("RH", "ISBL", f"{p:04d}", f)) is not None})
            z = {}
            for p in config.Z_LEVELS:
                ms = msgs(f, "HGT", "ISBL", f"{p:04d}")
                if ms:
                    for m in ms:
                        m.values = thermo.to_gpm(m.values, p)             # MSC may publish decametres
                    z[p] = disp("HGT", "ISBL", f"{p:04d}", f)
            kw["z"] = common.profile(z)
            out_steps.append(common.finish_step(f, **{k: v for k, v in kw.items() if v is not None}))
        if out_steps and out_steps[-1].ps_hpa is None:
            notes.append("GEPS: surface pressure not available; every pressure level is treated as above ground")
        return EnsembleData("geps", cycle, members, out_steps, notes)
