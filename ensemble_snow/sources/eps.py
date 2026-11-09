"""ECMWF ENS ("EPS", 51 members) from ECMWF Open Data, through the official ``ecmwf-opendata`` client.

The 0.25 deg ENS files hold all members and parameters of one forecast hour:  ``<root>/YYYYMMDD/HHz/ifs/0p25/enfo/YYYYMMDDHH0000-<step>h-enfo-ef.grib2``
(00/12 UTC: steps 0-144 every 3 h, then every 6 h to 360).  Total precipitation (tp) and snowfall (sf, liquid equivalent) are accumulated since the start
of the forecast, in metres.  ``sf`` is used as the model's own snow-type liquid; without it the wet-bulb ramp is used.

The ``ecmwf-opendata`` client is imported lazily; pass ``retrieve=`` to substitute it (the tests do).
"""
from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from .. import config, thermo
from ..fields import EnsembleData
from ..grib import iter_file
from ..http import Fetcher, NotFound
from . import common

log = logging.getLogger(__name__)
ROOTS = {"ecmwf": "https://data.ecmwf.int/forecasts", "google": "https://storage.googleapis.com/ecmwf-open-data",
         "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"}
# ECMWF's own server accepts multi-range requests (one request per message group); the cloud mirrors take one request per message and are
# throttled much sooner ("503 SlowDown").  Sources are tried in this order, each request moving on to the next one when it keeps failing.
DEFAULT_SOURCES = "ecmwf,google,aws"
G = 9.80665

# The ECMWF mirror on S3 answers "503 SlowDown" when too many requests arrive at once, and the client's own default is to wait 120 s and retry up to
# 500 times, all threads in lock-step.  So: few requests in flight, a short jittered start, a bounded retry budget, and a second try for a failed step.
PARALLEL = int(os.environ.get("ENSNOW_ECMWF_PARALLEL", "3"))
CLIENT_RETRIES = int(os.environ.get("ENSNOW_ECMWF_RETRIES", "5"))
CLIENT_RETRY_AFTER = int(os.environ.get("ENSNOW_ECMWF_RETRY_AFTER", "20"))
_GATE = threading.BoundedSemaphore(PARALLEL)


def index_url(root: str, cycle: datetime, step: int) -> str:
    d, hh = f"{cycle:%Y%m%d}", f"{cycle:%H}"
    return f"{root}/{d}/{hh}z/ifs/0p25/enfo/{d}{hh}0000-{step}h-enfo-ef.index"


class EpsSource:
    key = "eps"

    def __init__(self, fetcher: Fetcher | None = None, source: str | None = None, retrieve=None):
        self.http = fetcher or Fetcher()
        names = (source or os.environ.get("ENSNOW_ECMWF_SOURCE") or DEFAULT_SOURCES).split(",")
        self.sources = [n.strip() for n in names if n.strip() in ROOTS] or ["aws"]
        self._retrieve = retrieve

    @property
    def root(self) -> str:
        return ROOTS[self.sources[0]]

    def is_complete(self, cycle: datetime, max_fhr: int) -> bool:
        """True when some source has the last step; that source is then tried first."""
        for name in list(self.sources):
            if self.http.exists(index_url(ROOTS[name], cycle, max_fhr)):
                self.sources.remove(name)
                self.sources.insert(0, name)
                return True
        return False

    # ------------------------------------------------------------------ what the files contain
    def probe(self, cycle: datetime, step: int) -> set[tuple[str, str]]:
        """{(param, levtype)} found in the file's JSON-lines index, or an empty set if the index cannot be read."""
        text = None
        for name in self.sources:
            try:
                text = self.http.get(index_url(ROOTS[name], cycle, step)).decode("utf-8", "replace")
                break
            except (NotFound, RuntimeError):
                continue
        if text is None:
            return set()
        found = set()
        for line in text.splitlines():
            try:
                r = json.loads(line)
                found.add((str(r["param"]), str(r["levtype"])))
            except (ValueError, KeyError):
                continue
        return found

    def _client_retrieve(self, source=None, **req):
        if self._retrieve is not None:
            return self._retrieve(**req)
        from ecmwf.opendata import Client
        with _GATE:
            time.sleep(random.uniform(0.0, 1.5))
            return Client(source=source or self.sources[0], model="ifs", resol="0p25", maximum_retries=CLIENT_RETRIES,
                          retry_after=CLIENT_RETRY_AFTER).retrieve(**req)

    def _get(self, cycle, step, levtype, params, levels=None):
        """Download one request to a temporary file and decode it.  Returns a list of messages ([] on failure)."""
        req = dict(date=f"{cycle:%Y%m%d}", time=cycle.hour, step=step, stream="enfo", type=["cf", "pf"], levtype=levtype, param=list(params))
        if levels:
            req["levelist"] = list(levels)
        for attempt in range(1, 4 + len(self.sources)):
            fd, path = tempfile.mkstemp(suffix=".grib2")
            os.close(fd)
            try:
                src = self.sources[(attempt - 1) % len(self.sources)]
                self._client_retrieve(source=src, target=path, **req)
                try:
                    msgs = list(iter_file(path))
                except Exception as exc:                  # the file arrived but cannot be read: another download would not help
                    log.error("EPS step %s %s: downloaded from %s but could not be decoded: %s", step, levtype, src, exc)
                    return []
                log.info("EPS step %3s %s: %d messages (from %s)", step, levtype, len(msgs), src)
                return msgs
            except Exception as exc:
                log.warning("EPS step %s %s %s failed (attempt %d): %s", step, levtype, params, attempt, exc)
                if self._retrieve is None:
                    time.sleep(random.uniform(10, 25))
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return []

    def _task(self, cycle, step, have: set[tuple[str, str]]):
        has = (lambda p, lt: not have or (p, lt) in have)
        sfc_params = [p for p in ("tp", "sf", "2t", "2d", "sp") if has(p, "sfc")]
        z_param = "gh" if has("gh", "pl") else ("z" if has("z", "pl") else None)
        sfc = self._get(cycle, step, "sfc", sfc_params) if sfc_params else []
        # one request per variable, each at only the levels the SLR methods use (t: 6, r: 5, height: 3 = 14 fields per member instead of 18)
        pl = []
        if step % config.EPS_PL_STEP_H == 0:             # upper-air fields are 3/4 of the download: fetched every 6 h, interpolated in between
            for param, levels in (("t", config.T_LEVELS), ("r", config.RH_LEVELS), (z_param, config.Z_LEVELS)):
                if param and has(param, "pl"):
                    pl += self._get(cycle, step, "pl", [param], sorted(levels))
        return step, sfc, pl

    def load(self, cycle: datetime, steps: list[int], workers: int = 6) -> EnsembleData:
        have = self.probe(cycle, steps[-1])
        by_step: dict[int, tuple[list, list]] = {}
        with ThreadPoolExecutor(min(workers, PARALLEL + 1)) as ex:
            for step, sfc, pl in ex.map(lambda s: self._task(cycle, s, have), steps):
                by_step[step] = (sfc, pl)

        def pick(msgs, short, level=None, tol_level=True):
            sel = [m for m in msgs if m.short_name == short and (level is None or m.level == level)]
            return sel

        first_tp = next((pick(by_step[s][0], "tp") for s in steps if s > 0 and pick(by_step[s][0], "tp")), None)
        if not first_tp:
            raise RuntimeError("EPS: no total precipitation found (check the cycle and source)")
        members = sorted(common.member_ids(first_tp))
        notes: list[str] = []

        def cumulative(short):
            tp_by_step = {s: pick(by_step[s][0], short) for s in steps if s > 0}
            tp_by_step = {s: v for s, v in tp_by_step.items() if v}
            return common.assemble_precip(tp_by_step, members, steps) if tp_by_step else {}

        precip = cumulative("tp")
        snow = cumulative("sf")
        if not snow:
            notes.append("ECMWF snowfall (sf) not found; precipitation type estimated from the 2 m wet-bulb temperature")

        def disp(msgs, short, level=None, scale=1.0, offset=0.0):
            sel = pick(msgs, short, level)
            if not sel:
                return None
            return common.to_display(common.stack(sel, members, scale, offset))

        out_steps = []
        for s in steps:
            sfc, pl = by_step[s]
            kw: dict = {}
            if s > 0:
                kw["precip_cum_mm"] = precip.get(s)
                kw["snow_liq_cum_mm"] = snow.get(s) if snow else None
            kw["t2m_k"] = disp(sfc, "2t")
            kw["td2m_k"] = disp(sfc, "2d")
            kw["ps_hpa"] = disp(sfc, "sp", scale=0.01)
            kw["t"] = common.profile({p: a for p in config.T_LEVELS if (a := disp(pl, "t", p)) is not None})
            kw["rh"] = common.profile({p: a for p in config.RH_LEVELS if (a := disp(pl, "r", p)) is not None})
            z = {p: a for p in config.Z_LEVELS if (a := disp(pl, "gh", p)) is not None}
            if not z:                                                      # geopotential (m2 s-2) instead of height
                z = {p: a for p in config.Z_LEVELS if (a := disp(pl, "z", p, scale=1.0 / G)) is not None}
            kw["z"] = common.profile(z)
            out_steps.append(common.finish_step(s, **{k: v for k, v in kw.items() if v is not None}))
        fetched = {s for s in steps if by_step[s][1]}
        if len(fetched) < len(steps):
            common.fill_profiles(out_steps, fetched)
            notes.append(f"ECMWF upper-air fields are used every {config.EPS_PL_STEP_H} h and interpolated in time for the 3-hourly steps in between")
        if out_steps and out_steps[-1].ps_hpa is None:
            notes.append("EPS: surface pressure not available; every pressure level is treated as above ground")
        return EnsembleData("eps", cycle, members, out_steps, notes)
