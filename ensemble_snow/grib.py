"""GRIB2 decoding with ecCodes: regular lat/lon grids only (all three systems publish that), cropped to the display region on the way in."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .accum import parse_step_range
from .regrid import crop_slices


@dataclass
class Msg:
    short_name: str
    type_of_level: str
    level: int
    step_range: tuple[int, int]
    member: int
    values: np.ndarray            # (nlat, nlon) cropped, latitude ascending, longitude -180..180 ascending, NaN = missing
    lat: np.ndarray
    lon: np.ndarray
    name: str = ""
    units: str = ""


_axes_cache: dict[tuple, tuple] = {}


def _get(ec, gid, key, default=None):
    try:
        if ec.codes_is_defined(gid, key):
            return ec.codes_get(gid, key)
    except Exception:
        pass
    return default


def _axes(ec, gid):
    key = tuple(_get(ec, gid, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
                                           "longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"))
    if key in _axes_cache:
        return _axes_cache[key]
    ni, nj, la1, la2, lo1, lo2 = key
    if _get(ec, gid, "gridType") not in ("regular_ll", "regular_gg") and ni is None:
        raise ValueError("only regular lat/lon GRIB grids are supported")
    lat = np.linspace(la1, la2, nj)
    lon = np.linspace(lo1, lo2, ni)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    flip_lat = lat[0] > lat[-1]
    lat_a = lat[::-1] if flip_lat else lat
    lon_order = np.argsort(lon)
    lon_a = lon[lon_order]
    ys, xs = crop_slices(lat_a, lon_a)
    out = (flip_lat, lon_order, ys, xs, lat_a[ys].astype(np.float64), lon_a[xs].astype(np.float64), nj, ni)
    _axes_cache[key] = out
    return out


def decode_message(gid, ec=None) -> Msg:
    import eccodes as ec_
    ec = ec or ec_
    flip_lat, lon_order, ys, xs, lat, lon, nj, ni = _axes(ec, gid)
    vals = np.asarray(ec.codes_get_values(gid), np.float32).reshape(nj, ni)
    missing = _get(ec, gid, "missingValue", 9999.0)
    vals = np.where(vals == missing, np.nan, vals)
    if flip_lat:
        vals = vals[::-1]
    vals = vals[:, lon_order][ys, xs]
    sr = str(_get(ec, gid, "stepRange", _get(ec, gid, "step", 0)))
    member = _get(ec, gid, "perturbationNumber", _get(ec, gid, "number", 0))
    return Msg(str(_get(ec, gid, "shortName", "")), str(_get(ec, gid, "typeOfLevel", "")), int(_get(ec, gid, "level", 0) or 0),
               parse_step_range(sr), int(member or 0), vals, lat, lon, str(_get(ec, gid, "name", "")), str(_get(ec, gid, "units", "")))


def iter_messages(data: bytes):
    """Decode every GRIB message in ``data`` (concatenated messages are fine)."""
    import eccodes as ec
    pos = 0
    n = len(data)
    while pos < n:
        j = data.find(b"GRIB", pos)
        if j < 0:
            return
        total = int.from_bytes(data[j + 8:j + 16], "big")          # GRIB2 section 0: total message length
        if total < 16:
            return
        gid = ec.codes_new_from_message(data[j:j + total])
        try:
            yield decode_message(gid, ec)
        finally:
            ec.codes_release(gid)
        pos = j + total


import threading

DECODE_LOCK = threading.Lock()          # ecCodes is not guaranteed thread-safe: downloads run in parallel, decoding is serialised


def iter_file(path):
    """Decode every message of a GRIB file on disk, one at a time (large ensemble files are never held in memory)."""
    import eccodes as ec
    with open(path, "rb") as f, DECODE_LOCK:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                return
            try:
                yield decode_message(gid, ec)
            finally:
                ec.codes_release(gid)
