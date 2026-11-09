"""NOAA GEFS (31 members) from the NOAA Open Data bucket on AWS, read with byte ranges guided by the ``.idx`` inventories.

Layout (verified against the NOAA / NODD documentation, not against the live bucket from the build sandbox):

    s3://noaa-gefs-pds/gefs.YYYYMMDD/HH/atmos/pgrb2sp25/gep01.tHHz.pgrb2s.0p25.fFFF   0.25 deg surface fields (APCP, CSNOW, T2m, DPT, PRES ...)
    s3://noaa-gefs-pds/gefs.YYYYMMDD/HH/atmos/pgrb2ap5/gep01.tHHz.pgrb2a.0p50.fFFF    0.5 deg upper-air fields (TMP, RH, HGT on pressure levels)

``gec00`` is the control, ``gep01``..``gep30`` the perturbed members.  APCP is accumulated in buckets that restart every 6 hours; the GRIB step range
("6-12") is used to turn them into totals since initialisation.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from .. import config, thermo
from ..fields import EnsembleData
from ..grib import DECODE_LOCK, Msg, iter_messages
from ..http import Fetcher, NotFound
from . import common

log = logging.getLogger(__name__)
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
SURFACE_WANT = {("APCP", "surface"), ("CSNOW", "surface"), ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
                ("RH", "2 m above ground"), ("PRES", "surface")}


def member_name(m: int) -> str:
    return "gec00" if m == 0 else f"gep{m:02d}"


def url(cycle: datetime, kind: str, member: int, fhr: int) -> str:
    d, hh = f"{cycle:%Y%m%d}", f"{cycle:%H}"
    if kind == "s":
        return f"{BASE}/gefs.{d}/{hh}/atmos/pgrb2sp25/{member_name(member)}.t{hh}z.pgrb2s.0p25.f{fhr:03d}"
    return f"{BASE}/gefs.{d}/{hh}/atmos/pgrb2ap5/{member_name(member)}.t{hh}z.pgrb2a.0p50.f{fhr:03d}"


def parse_idx(text: str) -> list[dict]:
    """wgrib2-style inventory: ``num:offset:d=YYYYMMDDHH:VAR:level:fcst[:...]`` -> list of dicts with the byte range of each record."""
    recs = []
    for line in text.splitlines():
        p = line.split(":")
        if len(p) < 6:
            continue
        try:
            recs.append({"num": int(p[0]), "offset": int(p[1]), "var": p[3], "level": p[4], "fcst": p[5]})
        except ValueError:
            continue
    for a, b in zip(recs, recs[1:]):
        a["end"] = b["offset"] - 1
    if recs:
        recs[-1]["end"] = None
    return recs


def select(recs: list[dict], want) -> list[dict]:
    return [r for r in recs if want(r["var"], r["level"])]


def coalesce(recs: list[dict]) -> list[tuple[int, int | None, int]]:
    """Merge records whose byte ranges touch: -> [(start, end, number of records)]."""
    out: list[list] = []
    for r in sorted(recs, key=lambda r: r["offset"]):
        if out and out[-1][1] is not None and out[-1][1] + 1 == r["offset"]:
            out[-1][1], out[-1][2] = r["end"], out[-1][2] + 1
        else:
            out.append([r["offset"], r["end"], 1])
    return [tuple(o) for o in out]


def upper_want(var: str, level: str) -> bool:
    m = re.fullmatch(r"(\d+) mb", level)
    if not m:
        return False
    p = int(m.group(1))
    return (var == "TMP" and p in config.T_LEVELS) or (var == "RH" and p in config.RH_LEVELS) or (var == "HGT" and p in config.Z_LEVELS)


class GefsSource:
    key = "gefs"

    def __init__(self, fetcher: Fetcher | None = None):
        self.http = fetcher or Fetcher()
        self.n_members = config.ENSEMBLES["gefs"].n_members

    def is_complete(self, cycle: datetime, max_fhr: int) -> bool:
        return self.http.exists(url(cycle, "s", self.n_members - 1, max_fhr) + ".idx")

    # ------------------------------------------------------------------ one file
    def _read(self, u: str, want) -> list[tuple[dict, Msg]]:
        """Fetch the wanted records of one GRIB file.  Returns [(idx record, decoded message)]."""
        try:
            recs = select(parse_idx(self.http.get(u + ".idx").decode("ascii", "replace")), lambda v, l: want(v, l))
        except NotFound:
            return []
        out = []
        ordered = sorted(recs, key=lambda r: r["offset"])
        k = 0
        for start, end, n in coalesce(ordered):
            data = self.http.get(u, (start, end))
            group = ordered[k:k + n]
            k += n
            with DECODE_LOCK:
                msgs = list(iter_messages(data))
            if len(msgs) != len(group):
                log.warning("%s: expected %d messages in bytes %s-%s, decoded %d", u, len(group), start, end, len(msgs))
                continue
            out.extend(zip(group, msgs))
        return out

    def _task(self, cycle: datetime, member: int, fhr: int):
        sfc = self._read(url(cycle, "s", member, fhr), lambda v, l: (v, l) in SURFACE_WANT)
        upper = self._read(url(cycle, "a", member, fhr), upper_want)
        return member, fhr, sfc, upper

    # ------------------------------------------------------------------ everything
    def load(self, cycle: datetime, steps: list[int], workers: int = 16) -> EnsembleData:
        from collections import defaultdict
        from ..accum import CumulativeTracker

        members = list(range(self.n_members))
        jobs = [(m, f) for f in steps for m in members]
        found: dict[tuple[str, str, int], dict[int, Msg]] = defaultdict(dict)       # (var, level, fhr) -> {member: message}
        with ThreadPoolExecutor(workers) as ex:
            for m, f, sfc, upper in ex.map(lambda mf: self._task(cycle, *mf), jobs):
                for rec, msg in sfc + upper:
                    found[(rec["var"], rec["level"], f)][m] = msg

        # precipitation: each member's windows ("0-6", "6-12", ...) resolved to totals since initialisation
        cum: dict[int, dict[int, np.ndarray]] = {m: {} for m in members}
        axes = None
        for m in members:
            tr = CumulativeTracker()
            for f in steps:
                msg = found.get(("APCP", "surface", f), {}).get(m)
                if f > 0 and msg is not None:
                    a, b = msg.step_range
                    tr.add(a if b > a else 0, b if b > a else f, msg.values * common.water_scale(msg))
                    axes = (msg.lat, msg.lon)
            cum[m] = {f: v for f, v in tr.cum.items() if f > 0 and f in steps and not isinstance(v, float)}
        have = [m for m in members if cum[m]]
        if not have or axes is None:
            raise RuntimeError("GEFS: no precipitation found (check the cycle / bucket layout)")
        notes = []
        if len(have) < len(members):
            notes.append(f"{len(members) - len(have)} GEFS member(s) were incomplete and are left out")
        members = have
        rg = common.regridder_for(*axes)

        def get(var, level, f, scale=1.0):
            d = found.get((var, level, f))
            if not d:
                return None
            arr, la, lo = common.stack_dict(d, members, scale)
            return common.regridder_for(la, lo)(arr)

        out_steps = []
        for f in steps:
            kw: dict = {}
            if f > 0:
                nan = np.full((len(axes[0]), len(axes[1])), np.nan, np.float32)
                kw["precip_cum_mm"] = rg(np.stack([cum[m].get(f, nan) for m in members]))
                kw["snow_frac"] = get("CSNOW", "surface", f)
            kw["t2m_k"] = get("TMP", "2 m above ground", f)
            kw["td2m_k"] = get("DPT", "2 m above ground", f)
            kw["rh2m"] = get("RH", "2 m above ground", f)
            kw["ps_hpa"] = get("PRES", "surface", f, 0.01)
            for key, var, levels in (("t", "TMP", config.T_LEVELS), ("rh", "RH", config.RH_LEVELS), ("z", "HGT", config.Z_LEVELS)):
                kw[key] = common.profile({p: a for p in levels if (a := get(var, f"{p} mb", f)) is not None})
            out_steps.append(common.finish_step(f, **{k: v for k, v in kw.items() if v is not None}))
        if not any(s.snow_frac is not None for s in out_steps):
            notes.append("GEFS: no categorical snow flag found; precipitation type estimated from the 2 m wet-bulb temperature")
        return EnsembleData("gefs", cycle, members, out_steps, notes)
