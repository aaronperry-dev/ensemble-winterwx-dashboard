"""Helpers shared by the real sources: turning decoded messages into :class:`StepFields` on the display grid."""
from __future__ import annotations

import logging

import numpy as np

from .. import thermo
from ..accum import CumulativeTracker, MissingWindow
from ..fields import Profile, StepFields
from ..grib import Msg
from ..regrid import Regridder

log = logging.getLogger(__name__)
_regridders: dict[tuple, Regridder] = {}


def regridder_for(lat: np.ndarray, lon: np.ndarray) -> Regridder:
    key = (len(lat), len(lon), round(float(lat[0]), 4), round(float(lat[-1]), 4), round(float(lon[0]), 4), round(float(lon[-1]), 4))
    if key not in _regridders:
        _regridders[key] = Regridder(lat, lon)
    return _regridders[key]


def member_ids(msgs: list[Msg]) -> list[int]:
    """Member numbers of the messages of one variable.  Falls back to message order if the numbers are not unique."""
    ids = [m.member for m in msgs]
    return ids if len(set(ids)) == len(ids) else list(range(len(msgs)))


def stack(msgs: list[Msg], members: list[int], scale: float = 1.0, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M, nlat, nlon) array in the order of ``members`` (NaN for a member this variable lacks), plus the source lat/lon axes."""
    ids = member_ids(msgs)
    by = {i: m for i, m in zip(ids, msgs)}
    ref = msgs[0]
    out = np.full((len(members),) + ref.values.shape, np.nan, np.float32)
    for k, mem in enumerate(members):
        if mem in by:
            out[k] = by[mem].values * scale + offset
    return out, ref.lat, ref.lon


def to_display(arr_lat_lon: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    arr, lat, lon = arr_lat_lon
    return regridder_for(lat, lon)(arr)


def water_scale(msg: Msg) -> float:
    """Factor that turns a precipitation message into millimetres of liquid (ECMWF publishes metres, NCEP and MSC kg m-2)."""
    u = msg.units.replace(" ", "").lower()
    return 1000.0 if u in ("m", "m**1") else 1.0


def resolve_accumulation(msgs_by_step: dict[int, list[Msg]], members: list[int], steps: list[int]) -> dict[int, np.ndarray] | None:
    """Total-since-initialisation (mm, native grid) per step from messages that may carry bucketed windows."""
    out: dict[int, np.ndarray] = {}
    trackers = None
    src = None
    for f in sorted(msgs_by_step):
        msgs = msgs_by_step[f]
        if not msgs:
            continue
        arr, lat, lon = stack(msgs, members, water_scale(msgs[0]))
        src = (lat, lon)
        a, b = msgs[0].step_range
        if trackers is None:
            trackers = CumulativeTracker()
        trackers.add(a if b > a else 0, b if b > a else f, arr)
        out[f] = (arr, lat, lon)
    if trackers is None:
        return None
    res = {}
    for f in sorted(out):
        try:
            res[f] = trackers.get(f)
        except MissingWindow:
            log.warning("accumulation for hour %s cannot be resolved (missing earlier window)", f)
    return {"series": res, "axes": src}


def assemble_precip(msgs_by_step: dict[int, list[Msg]], members: list[int], steps: list[int]):
    """-> {fhr: display-grid cumulative precipitation (M, ny, nx)} or {} when nothing usable."""
    r = resolve_accumulation(msgs_by_step, members, steps)
    if not r:
        return {}
    lat, lon = r["axes"]
    rg = regridder_for(lat, lon)
    return {f: rg(v) for f, v in r["series"].items() if not isinstance(v, float)}


def finish_step(fhr: int, **kw) -> StepFields:
    """Build a StepFields, converting units on the way: temperatures K, humidity %, ps hPa."""
    sf = StepFields(fhr)
    for k, v in kw.items():
        setattr(sf, k, v)
    if sf.rh2m is None and sf.t2m_k is not None and sf.td2m_k is not None:
        sf.rh2m = thermo.rh_from_td(sf.t2m_k - 273.15, sf.td2m_k - 273.15).astype(np.float32)
    return sf


def profile(levels_arrays: dict[int, np.ndarray]) -> Profile | None:
    if not levels_arrays:
        return None
    lv = tuple(sorted(levels_arrays, reverse=True))
    return Profile(lv, np.stack([levels_arrays[l] for l in lv]))


def stack_dict(d: dict[int, Msg], members: list[int], scale: float = 1.0, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`stack`, for messages whose member number is known from where they came from (one file per member)."""
    ref = next(iter(d.values()))
    out = np.full((len(members),) + ref.values.shape, np.nan, np.float32)
    for k, mem in enumerate(members):
        if mem in d:
            out[k] = d[mem].values * scale + offset
    return out, ref.lat, ref.lon
