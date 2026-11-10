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
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config, thermo
from ..accum import CumulativeTracker
from ..fields import EnsembleData
from ..grib import DECODE_LOCK, Msg, iter_messages
from ..http import Fetcher, HostDown, NotFound
from ..parts import PartCache
from . import common

log = logging.getLogger(__name__)
# The same files are published by MSC on more than one server.  GitHub's runners are sometimes refused by dd.weather.gc.ca (connections time out),
# so each file is tried on the next server when one does not answer.  ``{date}`` is the run's UTC date.
MIRRORS = (
    "https://dd.weather.gc.ca/today/ensemble/geps/grib2/raw",
    "https://hpfx.collab.science.gc.ca/{date}/WXO-DD/ensemble/geps/grib2/raw",
    "https://dd.weather.gc.ca/{date}/WXO-DD/ensemble/geps/grib2/raw",
)
BASE = MIRRORS[0]
OPTIONAL = {"PRES"}                      # a file that may legitimately be absent: not worth asking the other servers for


def url(cycle: datetime, var: str, ltype: str, level: str, fhr: int, base: str = BASE) -> str:
    stamp = f"{cycle:%Y%m%d%H}"
    base = base.format(date=f"{cycle:%Y%m%d}")
    return f"{base}/{cycle:%H}/{fhr:03d}/CMC_geps-raw_{var}_{ltype}_{level}_latlon0p5x0p5_{stamp}_P{fhr:03d}_allmbrs.grib2"


def snow_is_water_equivalent(snow_cum: np.ndarray, precip_cum: np.ndarray) -> bool:
    """ASNOW looks like liquid-equivalent snow if it never exceeds the total precipitation by more than rounding (a depth would be ~10x larger)."""
    ok = precip_cum >= 2.0
    if ok.sum() < 5:
        return True
    ratio = snow_cum[ok] / precip_cum[ok]
    return bool(np.nanpercentile(ratio, 99) <= 1.2)


class GepsSource:
    key = "geps"
    MAX_WORKERS = 4                      # MSC throttles heavy parallel use; a few connections are enough
    BUDGET_S = 1500.0                    # stop fetching after this long and go on with what has arrived
    PASSES = 3                           # hours that failed are retried this many times in all, after a pause
    PAUSE_S = 75.0

    def __init__(self, fetcher: Fetcher | None = None, mirrors: tuple[str, ...] = MIRRORS):
        self.http = fetcher or Fetcher()
        self.mirrors = tuple(mirrors)
        self.good = self.mirrors[0]      # the server that last worked: tried first
        self.part_root: Path | None = None   # set by the build: where finished forecast hours are kept between runs
        self.part_tag = ""

    # -- servers --------------------------------------------------------------------------------------------------------------------
    def _order(self, cycle: datetime) -> list[str]:
        """Servers to try, the last one that worked first; one that is known to be down goes to the end."""
        ranked = [self.good] + [m for m in self.mirrors if m != self.good]
        probe = lambda m: url(cycle, "APCP", "SFC", "0", 6, m)
        up = [m for m in ranked if not self._down(probe(m))]
        return up + [m for m in ranked if m not in up]

    def _down(self, u: str) -> bool:
        check = getattr(self.http, "host_down", None)
        return bool(check and check(u))

    def is_complete(self, cycle: datetime, max_fhr: int) -> bool:
        for m in self._order(cycle):
            if self.http.exists(url(cycle, "APCP", "SFC", "0", max_fhr, m)):
                self.good = m
                return True
        return False

    def _fetch(self, cycle, var, ltype, level, fhr) -> list[Msg] | None:
        """The decoded messages of one file; None when it does not exist; an exception when no server could deliver it."""
        err: Exception | None = None
        absent = False
        for m in self._order(cycle):
            try:
                data = self.http.get(url(cycle, var, ltype, level, fhr, m))
            except NotFound:
                absent = True
                if var in OPTIONAL:
                    return None
                continue
            except Exception as exc:                  # host down, or the retries ran out: try the next server
                err = exc
                continue
            if m != self.good:
                log.warning("GEPS: switching to %s", m.split("/")[2])
                self.good = m
            with DECODE_LOCK:
                return list(iter_messages(data))
        if err is not None:
            raise err
        return None if absent else None

    def _task(self, cycle, fhr):
        """All files of one forecast hour.  Returns (hour, {file: messages}, number of files that could not be fetched)."""
        want = [("APCP", "SFC", "0"), ("ASNOW", "SFC", "0"), ("TMP", "TGL", "2m"), ("RH", "TGL", "2m"), ("PRES", "SFC", "0")]
        want += [("TMP", "ISBL", f"{p:04d}") for p in config.T_LEVELS]
        want += [("RH", "ISBL", f"{p:04d}") for p in config.RH_LEVELS]
        want += [("HGT", "ISBL", f"{p:04d}") for p in config.Z_LEVELS]
        out, failed = {}, 0
        for var, lt, lev in want:
            if fhr == 0 and var in ("APCP", "ASNOW"):
                continue
            try:
                out[(var, lt, lev)] = self._fetch(cycle, var, lt, lev, fhr)
            except Exception as exc:
                failed += 1
                log.debug("GEPS F%03d %s %s %s: %s", fhr, var, lt, lev, exc)
                if isinstance(exc, HostDown) or failed >= 3:       # no use going on with this hour now; it is tried again later
                    break
        return fhr, out, failed

    def _parts(self, cycle: datetime) -> PartCache | None:
        if self.part_root is None:
            return None
        try:
            return PartCache(self.part_root, "geps", f"{self.part_tag}_{cycle:%Y%m%d%H}")
        except Exception:
            log.exception("GEPS: no place to keep the forecast hours already downloaded (the build goes on)")
            return None

    def _download(self, cycle: datetime, steps: list[int], workers: int) -> tuple[dict[int, dict], list[int]]:
        """Every forecast hour that can be had: those already kept on disk, then the rest; hours that fail are retried after a pause.
        Returns the files per hour and the hours still missing."""
        workers = max(1, min(workers, self.MAX_WORKERS))
        parts = self._parts(cycle)
        deadline = time.monotonic() + self.BUDGET_S
        data: dict[int, dict] = {}
        todo = list(steps)
        if parts:
            for f in steps:
                got = parts.get(f)
                if got is not None:
                    data[f] = got
            todo = [f for f in steps if f not in data]
            if data:
                log.info("GEPS: %d of %d forecast hours were kept from an earlier attempt", len(data), len(steps))

        def one(f):
            if time.monotonic() > deadline:
                return f, {}, 1
            return self._task(cycle, f)

        for attempt in range(self.PASSES):
            if not todo:
                break
            if attempt:
                if time.monotonic() + self.PAUSE_S > deadline:
                    break
                log.warning("GEPS: %d forecast hours failed; trying them again in %.0f s", len(todo), self.PAUSE_S)
                time.sleep(self.PAUSE_S)
            still = []
            with ThreadPoolExecutor(workers) as ex:
                for fhr, out, failed in ex.map(one, todo):
                    if out or fhr not in data:
                        data[fhr] = {**data.get(fhr, {}), **out}
                    if failed:
                        still.append(fhr)
                    elif parts:
                        parts.put(fhr, out)
            todo = still
        return data, todo

    def load(self, cycle: datetime, steps: list[int], workers: int = 8) -> EnsembleData:
        data, missing = self._download(cycle, steps, workers)
        if missing:
            log.warning("GEPS: %d of %d forecast hours could not be downloaded (%s)", len(missing), len(steps),
                        ", ".join(f"F{f:03d}" for f in missing[:12]) + (" ..." if len(missing) > 12 else ""))
            if not any(f > 0 and data.get(f, {}).get(("APCP", "SFC", "0")) for f in steps):
                raise RuntimeError(f"GEPS: nothing could be downloaded ({self.good.split('/')[2]} and its mirrors are not answering)")

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
            both = set(snow) & set(precip)
            use_snow = bool(both) and snow_is_water_equivalent(snow[max(both)], precip[max(both)])
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
        if missing:
            notes.append("GEPS download incomplete: forecast hour(s) " + ", ".join(f"F+{f}" for f in missing[:12]) + (" ..." if len(missing) > 12 else "") +
                         " could not be fully fetched from the Meteorological Service of Canada; they are filled in from neighbouring hours where possible")
        return EnsembleData("geps", cycle, members, out_steps, notes, partial=bool(missing))
