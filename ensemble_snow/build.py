"""Build the static site:  fetch -> snowfall by every method -> ensemble statistics -> compact files + manifest -> copy the page.

    python -m ensemble_snow.build --out public                    # newest complete run of every system
    python -m ensemble_snow.build --out public --synthetic        # demo data, no network (the page says so)
    python -m ensemble_snow.build --probe                         # check which runs / files are reachable, change nothing

No data are ever committed: GitHub Actions runs this, uploads ``public/`` as a Pages artifact and deploys it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import compute, config, consensus, encode, gis, slr, stats, terrain
from .fields import EnsembleData
from .sources.base import forecast_steps

log = logging.getLogger("ensemble_snow")
STAT_NAMES = list(config.ALL_STATS)
BLEND_LABEL = "Multi-method mean"


# --------------------------------------------------------------------------- cycle selection
def candidate_cycles(now: datetime, hours: tuple[int, ...], n: int = 8) -> list[datetime]:
    """Cycle times at the given hours not later than ``now``, newest first."""
    t = now.replace(minute=0, second=0, microsecond=0)
    out = []
    while len(out) < n:
        if t.hour in hours and t <= now:
            out.append(t)
        t -= timedelta(hours=1)
    return out


def pick_cycles(sources: dict, now: datetime, max_fhr: int, common: bool = True, force: datetime | None = None) -> dict[str, datetime]:
    """The run each system is built from.  A system with no complete run at all is left out (the others carry on); when the systems cannot share
    one initialisation time each uses its own newest complete run and the combined view lines them up by valid time."""
    if force is not None:
        return {k: force for k in sources}
    per = {k: candidate_cycles(now, config.ENSEMBLES[k].cycles) for k in sources}
    memo: dict[tuple[str, datetime], bool] = {}

    def complete(k, t):
        if (k, t) not in memo:
            try:
                memo[k, t] = bool(sources[k].is_complete(t, max_fhr))
            except Exception as exc:                          # a server that cannot be reached counts as "not there yet"
                log.warning("%s: could not check the run of %s (%s)", k, t, exc)
                memo[k, t] = False
        return memo[k, t]

    own: dict[str, datetime] = {}
    for k in sources:
        for t in per[k]:
            if complete(k, t):
                own[k] = t
                break
        else:
            log.error("%s: no complete run found in the last %d cycles; it is left out", k, len(per[k]))
    if common and len(own) > 1:
        shared = sorted(set.intersection(*[set(per[k]) for k in own]), reverse=True)
        newest = max(own.values())
        for t in shared[:6]:
            if all(complete(k, t) for k in own):
                if newest - t <= timedelta(hours=12):
                    return {k: t for k in own}
                break
        log.warning("no recent run is complete for every system; each system's own newest run is used and the combined view lines them up by valid time")
    return own


# --------------------------------------------------------------------------- statistics -> bundle
# --------------------------------------------------------------------------- parallel work
# The heavy steps (spatial alignment per accumulation window, then one bundle per snow-to-liquid method) are independent of each other, so they
# run in forked worker processes that inherit the arrays through ``_CTX`` instead of receiving copies.  ENSNOW_PROCESSES=1 runs everything serially.
_CTX: dict = {}


def _processes() -> int:
    try:
        n = int(os.environ.get("ENSNOW_PROCESSES", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else min(4, os.cpu_count() or 1)


def _pmap(func, items: list, ctx: dict) -> list:
    n = min(_processes(), len(items))
    if n <= 1 or "fork" not in mp.get_all_start_methods():
        _CTX.clear()
        _CTX.update(ctx)
        return [func(i) for i in items]
    _CTX.clear()
    _CTX.update(ctx)
    with ProcessPoolExecutor(n, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(func, items))


def _shift_job(window: int) -> np.ndarray:
    c = _CTX
    xw = stats.windowed(c["cum"], window, c["fhrs"])                           # (T, M, ny, nx)
    return np.stack([consensus.consensus_shifts(xw[t], c["weights"]) for t in range(len(c["fhrs"]))])


def compute_shifts(cum: np.ndarray, fhrs: list[int], weights: np.ndarray | None) -> dict[int, np.ndarray]:
    """Spatial-alignment displacement fields of the members, per accumulation window and time: ``{window: (T, M, 2, ny, nx)}``.

    Found once, on the multi-method snowfall, and applied to every method's members (the position error of a storm does not depend on the
    snow-to-liquid ratio), which keeps the cost of SAM independent of the number of methods."""
    res = _pmap(_shift_job, list(config.WINDOWS_H), dict(cum=cum, fhrs=fhrs, weights=weights))
    return dict(zip(config.WINDOWS_H, res))


QPF = "qpf"          # the key of the total-liquid-precipitation bundle (it is not a snow-to-liquid method)


def _bundle_job(key: str) -> tuple[str, int]:
    c = _CTX
    if key == QPF:
        return key, write_bundle(c["dir"] / c["names"][key], c["cum"][key], None, c["fhrs"], c["meta"][key], c["weights"], c["qpf_shifts"], parameter=QPF)
    return key, write_bundle(c["dir"] / c["names"][key], c["cum"][key], c["liquid"], c["fhrs"], c["meta"][key], c["weights"], c["shifts"])


def write_bundles(out_dir: Path, names: dict[str, str], cum: dict[str, np.ndarray], liquid, fhrs, meta: dict[str, dict], weights, shifts,
                  qpf_shifts=None) -> dict[str, int]:
    """One bundle per method (and one for total liquid precipitation, key ``"qpf"``), in parallel.  ``cum``: key -> (T, M, ny, nx).  Returns the size of each file."""
    done = _pmap(_bundle_job, list(names), dict(dir=out_dir, names=names, cum=cum, liquid=liquid, fhrs=fhrs, meta=meta, weights=weights, shifts=shifts,
                                                 qpf_shifts=qpf_shifts))
    return dict(done)


def _blocks(cum: np.ndarray, liquid: np.ndarray | None, fhrs: list[int], weights: np.ndarray | None, shifts: dict[int, np.ndarray], thresholds=config.THRESHOLDS_IN):
    """cum, liquid: (T, M, ny, nx).  -> amt (W,S,T,ny,nx), prob (W,K,T,ny,nx), slr (T,ny,nx) (None without ``liquid``).  S = percentile statistics, then PMM / LPMM / SAM / SAM-LPM."""
    amt, prob = [], []
    T, _, ny, nx = cum.shape
    for w in config.WINDOWS_H:
        xw = stats.windowed(cum, w, fhrs)                                    # (T, M, ny, nx)
        a, p = stats.ensemble_stats(np.moveaxis(xw, 1, 0), weights, thresholds)           # (S, T, ny, nx), (K, T, ny, nx)
        cons = np.empty((len(config.CONSENSUS_STATS), T, ny, nx), np.float32)
        for t in range(T):
            r = consensus.products(xw[t], shifts[w][t], weights)
            for i, k in enumerate(config.CONSENSUS_STATS):
                cons[i, t] = r[k]
        amt.append(np.concatenate([a, cons]))
        prob.append(p)
    if liquid is None:
        return np.stack(amt), np.stack(prob), None
    eff = np.moveaxis(compute.effective_ratio(cum, liquid), 1, 0)           # (M, T, ny, nx)
    if weights is None:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                slr_mean = np.nanmean(eff, axis=0)
    else:
        w = np.asarray(weights, np.float64).reshape(-1, 1, 1, 1) * ~np.isnan(eff)
        with np.errstate(all="ignore"):
            slr_mean = np.where(w.sum(0) > 0, (np.nan_to_num(eff) * w).sum(0) / w.sum(0), np.nan)
    return np.stack(amt), np.stack(prob), slr_mean.astype(np.float32)


def write_bundle(path: Path, cum, liquid, fhrs, meta: dict, weights=None, shifts=None, parameter: str = "snow") -> int:
    if shifts is None:
        shifts = compute_shifts(cum, fhrs, weights)
    if parameter == QPF:
        amt, prob, _ = _blocks(cum, None, fhrs, weights, shifts, config.QPF_THRESHOLDS_IN)
        return encode.write_bundle(path, [
            dict(name="amt", data=amt, dtype="u16", lo=0.0, step=config.QPF_STEP_IN),
            dict(name="prob", data=prob, dtype="u8", lo=0.0, step=1.0),
        ], meta)
    amt, prob, slr_mean = _blocks(cum, liquid, fhrs, weights, shifts)
    return encode.write_bundle(path, [
        dict(name="amt", data=amt, dtype="u16", lo=0.0, step=config.AMT_STEP_IN),
        dict(name="prob", data=prob, dtype="u8", lo=0.0, step=1.0),
        dict(name="slr", data=slr_mean, dtype="u8", lo=0.0, step=config.SLR_STEP),
    ], meta)


def bundle_meta(ensemble: str, method: str, init: str, fhrs: list[int]) -> dict:
    qpf = method == QPF
    return {"ensemble": ensemble, "method": method, "init": init, "fhrs": fhrs, "stats": STAT_NAMES,
            "thresholds": list(config.QPF_THRESHOLDS_IN if qpf else config.THRESHOLDS_IN), "windows": list(config.WINDOWS_H), "parameter": "qpf" if qpf else "snow"}


def window_times(fhrs: list[int]) -> dict[str, list[int]]:
    """Forecast hours at which each accumulation window is complete, for the page's time slider."""
    return {str(w): [f for f, ok in zip(fhrs, stats.window_valid(w, fhrs)) if ok] for w in config.WINDOWS_H}


def method_info() -> dict:
    info = {"blend": {"label": BLEND_LABEL, "group": "blend", "desc": "Mean of the ratios of the blended methods (" + ", ".join(
        slr.METHODS[k].label for k in config.BLEND if k in slr.METHODS) + ") applied to the same snow-type liquid. Methods a system cannot compute are left out.",
        "ref": "", "in_blend": False}}
    for k, m in slr.METHODS.items():
        info[k] = {"label": m.label, "group": m.group, "desc": m.desc, "ref": m.ref, "in_blend": m.in_blend}
    return info


def describe_input(ens: EnsembleData) -> None:
    """One log line per system saying what the download contains, so an empty field is obvious at a glance."""
    def top(attr):
        v = [getattr(s, attr) for s in ens.steps if getattr(s, attr) is not None]
        if not v:
            return "none"
        a = np.nanmax(v[-1]) if v else float("nan")
        return f"max {a:.1f} mm in {int(np.sum(np.nanmax(v[-1], axis=0) > 0.1)) if np.ndim(v[-1]) == 3 else 0} grid points"
    log.info("%s input at F+%d h: %d members, total precipitation %s, model snowfall %s, 2 m temperature %s; type from: %s", ens.key, ens.steps[-1].fhr,
             len(ens.members), top("precip_cum_mm"), top("snow_liq_cum_mm"), "present" if ens.steps[-1].t2m_k is not None else "MISSING", ens.ptype_source or "n/a")


def process_ensemble(ens: EnsembleData, out: Path, grid_lat, grid_lon, keep: dict | None = None) -> dict:
    spec = config.ENSEMBLES[ens.key]
    describe_input(ens)
    LON, LAT = np.meshgrid(grid_lon, grid_lat)
    res = compute.compute_snowfall(ens, LAT, LON)
    log.info("%s result: highest member snowfall by F+%d h (blend) %.1f in, %d of %d grid points above 0.1 in", ens.key, res.fhrs[-1],
             float(np.nanmax(res.methods["blend"][-1])), int(np.sum(np.nanmax(res.methods["blend"][-1], axis=0) > 0.1)), LAT.size)
    (out / "data").mkdir(parents=True, exist_ok=True)
    files = {}
    order = ["blend"] + [k for k in slr.METHODS if k in res.methods]
    t0 = time.time()
    shifts = compute_shifts(res.methods["blend"], res.fhrs, None)
    log.info("%s: spatial alignment of the members done in %.0f s", ens.key, time.time() - t0)
    init = ens.cycle.strftime("%Y-%m-%dT%H:%M:%SZ")
    qpf_shifts = compute_shifts(res.qpf_in, res.fhrs, None)
    names = {k: f"{ens.key}_{k}.bin.gz" for k in order + [QPF]}
    sizes = write_bundles(out / "data", names, {**res.methods, QPF: res.qpf_in}, res.liquid_in, res.fhrs,
                          {k: bundle_meta(ens.key, k, init, res.fhrs) for k in order + [QPF]}, None, shifts, qpf_shifts)
    for k in order + [QPF]:
        files[k] = {"file": names[k], "bytes": sizes[k]}
        log.info("%s/%s  %.0f kB", ens.key, k, sizes[k] / 1e3)
    if keep is not None:
        keep[ens.key] = (ens, res)
    notes = list(res.notes)
    return {"label": spec.label, "short": spec.short, "center": spec.center, "native": f"{spec.native}, {len(ens.members)} members", "license": spec.license, "url": spec.url,
            "init": ens.cycle.strftime("%Y-%m-%dT%H:%M:%SZ"), "members": len(ens.members), "fhrs": res.fhrs, "methods": order,
            "blend_members": res.blend_members, "ptype": res.ptype_source, "notes": notes, "files": files,
            "window_times": window_times(res.fhrs)}


def _aligned(ens, res, ref_init: datetime):
    """How one system's run lines up with the newest run: ``(lag in hours, {hour since ref_init: index into res.fhrs})``, or None when it cannot be used.

    A run older than the reference is moved by its age; its cumulative totals are then reduced by what had fallen by ``ref_init`` (in :func:`combine`)."""
    lag = int(round((ref_init - ens.cycle).total_seconds() / 3600))
    if lag < 0:
        return None
    if lag > config.BLEND_MAX_LAG_H:
        log.warning("%s: its run is %d h older than the newest, more than %d h: left out of the combined view", ens.key, lag, config.BLEND_MAX_LAG_H)
        return None
    if lag and lag not in res.fhrs:
        log.warning("%s: no output at the reference time (F+%d h of its run): left out of the combined view", ens.key, lag)
        return None
    at = {f - lag: i for i, f in enumerate(res.fhrs) if f - lag > 0}
    return lag, at


def parse_blend_weights(spec: str | None, keys=None) -> dict[str, float] | str:
    """The share each system has in the pooled view.  "" or "equal": every system counts the same (the default); "members": in proportion to
    its member count; "eps=50,gefs=30,geps=20": explicit relative weights (any positive numbers, they are normalised)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "equal", "members"):
        return spec or "equal"
    out: dict[str, float] = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in config.ENSEMBLES:
            raise ValueError(f"blend weights: unknown system {k!r} (use eps, gefs, geps)")
        try:
            out[k] = float(v)
        except ValueError:
            raise ValueError(f"blend weights: {part.strip()!r} is not system=number") from None
        if not out[k] > 0 or not np.isfinite(out[k]):
            raise ValueError(f"blend weights: the weight of {k} must be a positive number")
    missing = [k for k in (keys or []) if k not in out]
    if missing:
        raise ValueError(f"blend weights: give a weight for every system being built (missing {', '.join(missing)}), or use 'equal' or 'members'")
    return out


def system_shares(spec, systems: list[str], sizes: dict[str, int]) -> dict[str, float]:
    """Each system's share of the pool, summing to 1, over the systems actually in it (an absent system's weight is shared out by renormalising)."""
    if spec == "members":
        raw = {k: float(sizes[k]) for k in systems}
    elif isinstance(spec, dict):
        raw = {k: spec[k] for k in systems if k in spec}
        if len(raw) != len(systems):
            raise ValueError("blend weights: no weight given for " + ", ".join(k for k in systems if k not in spec))
    else:
        raw = {k: 1.0 for k in systems}
    tot = sum(raw.values())
    return {k: raw[k] / tot for k in systems}


def combine(kept: dict, out: Path, blend_weights="equal") -> dict | None:
    """Pool the systems into one ensemble.  By default every system carries equal total weight (``blend_weights`` can change the shares).

    Every system stays in the pool at every hour, so the pooled running totals only ever grow: a system (or member) that lacks an hour that
    another has is given the straight line between its neighbouring hours, and the pool ends where the earliest-ending system's data end.
    A method that a system cannot compute is pooled from the systems that can.  Runs of different ages are lined up by valid time."""
    systems = list(kept)
    if len(systems) < 2:
        return None
    ref = max(kept[k][0].cycle for k in systems)
    align = {}
    for k in systems:
        a = _aligned(kept[k][0], kept[k][1], ref)
        if a is not None:
            align[k] = a
    systems = [k for k in systems if k in align]
    if len(systems) < 2:
        log.info("combined view skipped: fewer than two systems can be lined up")
        return None
    ends = {k: max(align[k][1], default=0) for k in systems}
    longest = max(ends.values())
    short = [k for k in systems if ends[k] < config.BLEND_MIN_SHARE * longest]
    if short:                                    # a system that stops early would cut the pool short for everyone
        log.info("combined view: leaving out %s (data end at F+%s h, the longest at F+%d h)", short, [ends[k] for k in short], longest)
        systems = [k for k in systems if k not in short]
        if len(systems) < 2:
            log.info("combined view skipped: fewer than two systems of usable length")
            return None
        ends = {k: ends[k] for k in systems}
    last = min(ends.values())                    # beyond the earliest ending the pool would lose a system, and the running totals would jump
    every = sorted(set().union(*[set(align[k][1]) for k in systems]))
    fhrs = [f for f in every if f <= last]
    if not fhrs:
        log.info("combined view skipped: the systems have no hours in common")
        return None
    pos = {f: i for i, f in enumerate(every)}
    keep = len(fhrs)
    methods = [m for m in ["blend"] + list(slr.METHODS) if any(m in kept[k][1].methods for k in systems)]
    sizes = {k: len(kept[k][0].members) for k in systems}
    total = sum(sizes.values())
    start = dict(zip(systems, np.cumsum([0] + [sizes[k] for k in systems[:-1]])))
    ny, nx = kept[systems[0]][1].liquid_in.shape[2:]

    def pool(get):
        """Members x hours of running totals since the newest run's start.  A system without an hour that others have gets the straight line
        between its neighbouring hours (never extrapolated), so every system stays in the pool at every hour and the totals only ever grow."""
        arr = np.full((keep, total, ny, nx), np.nan, np.float32)
        for k in systems:
            lag, at = align[k]
            src = get(kept[k][1])
            if src is None:
                continue
            base = src[kept[k][1].fhrs.index(lag)] if lag else 0.0
            blk = np.full((len(every), sizes[k], ny, nx), np.nan, np.float32)
            for f, i in at.items():
                blk[pos[f]] = src[i] - base
            arr[:, start[k]:start[k] + sizes[k]] = compute.fill_time_gaps(blk, every, start_zero=True)[:keep]
        return arr

    pooled = {m: pool(lambda r, m=m: r.methods.get(m)) for m in methods}
    liq = pool(lambda r: r.liquid_in)
    qpf = pool(lambda r: r.qpf_in)
    shares = system_shares(blend_weights, systems, sizes)
    # a member's weight is its system's share spread over the system's members (scaled so equal shares give 1/size, as before)
    weights = np.concatenate([np.full(sizes[k], shares[k] * len(systems) / sizes[k]) for k in systems])
    equal = blend_weights == "equal"
    log.info("all: system weights %s", {k: f"{100 * v:.1f}%" for k, v in shares.items()})
    t0 = time.time()
    shifts = compute_shifts(pooled["blend"], fhrs, weights)
    qpf_shifts = compute_shifts(qpf, fhrs, weights)
    log.info("all: spatial alignment of the members done in %.0f s", time.time() - t0)
    init = ref.strftime("%Y-%m-%dT%H:%M:%SZ")
    names = {m: f"all_{m}.bin.gz" for m in methods + [QPF]}
    written = write_bundles(out / "data", names, {**pooled, QPF: qpf}, liq, fhrs, {m: bundle_meta("all", m, init, fhrs) for m in methods + [QPF]}, weights, shifts,
                            qpf_shifts)
    files = {m: {"file": names[m], "bytes": written[m]} for m in methods + [QPF]}
    contrib = [",".join(k for k in systems if f in align[k][1]) for f in fhrs]       # systems with data of their own at the hour
    filled = [",".join(k for k in systems if f not in align[k][1]) for f in fhrs]    # systems whose value at the hour is a straight line between neighbours
    shorts = " + ".join(config.ENSEMBLES[k].short for k in systems)
    pct = {k: f"{config.ENSEMBLES[k].short} {100 * shares[k]:.0f}%" for k in systems}
    if equal:
        notes = ["Every system is pooled at every hour with equal total weight, whatever its member count (a GEPS member counts for more than an EPS member)."]
    else:
        notes = [f"Systems are pooled at every hour with weights {', '.join(pct.values())} ("
                 + ("in proportion to member counts, so every member counts the same" if blend_weights == "members" else "set for this build") + "); members within a system count the same."]
    for k in systems:
        lag = align[k][0]
        if lag:
            notes.append(f"{config.ENSEMBLES[k].short} is from the run {lag} h before the newest; it is lined up by valid time and its snow is counted from the newest run's start.")
    for k in kept:
        if k in short:
            notes.append(f"{config.ENSEMBLES[k].short} is not in the combined view (its data end at F+{max(align[k][1], default=0)} h, far short of the others).")
        elif k not in systems:
            notes.append(f"{config.ENSEMBLES[k].short} is not in the combined view (its run is too old to line up).")
    if last < max(every):
        ended = " and ".join(config.ENSEMBLES[k].short for k in systems if ends[k] == last)
        notes.append(f"The combined view ends at F+{last} h, where {ended} data end; later hours are in each system's own view.")
    if any(filled):
        notes.append("Where a system has no data for an hour that another has, its totals are taken as a straight line between its neighbouring hours.")
    return {"label": "All systems combined", "short": "All", "center": " + ".join({"eps": "ECMWF", "gefs": "NOAA", "geps": "ECCC"}[k] for k in systems),
            "native": f"{total} members ({shorts}); " + ("each system carries equal weight" if equal else "weights " + ", ".join(pct.values())), "license": "see each system", "url": "",
            "init": init, "members": total, "fhrs": fhrs, "methods": methods, "blend_members": [],
            "ptype": "per system", "notes": notes, "files": files, "window_times": window_times(fhrs),
            "systems": [{"key": k, "short": config.ENSEMBLES[k].short, "init": kept[k][0].cycle.strftime("%Y-%m-%dT%H:%M:%SZ"), "members": sizes[k], "weight": round(shares[k], 4)} for k in systems],
            "contrib": contrib, "filled": filled}


# --------------------------------------------------------------------------- site
def _hash_files(root: Path, paths) -> str:
    import hashlib
    h = hashlib.sha1()
    for p in sorted(paths):
        if p.is_file() and "vendor" not in p.parts and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


_ROOT = Path(__file__).resolve().parent.parent


_NOT_MODEL_DATA = ("gis.py", "terrain.py")           # the basemap and the terrain file are made after, and apart from, the model data files


def pipeline_fingerprint() -> str:
    """Hash of the Python pipeline only.  The finished data files depend on nothing else (not on the page), so they are reused when it is unchanged."""
    return _hash_files(_ROOT, [p for p in (_ROOT / "ensemble_snow").rglob("*.py") if p.name not in _NOT_MODEL_DATA])


def code_fingerprint() -> str:
    """Hash of the pipeline and viewer sources: a code change rebuilds the site even when the model cycle has not changed."""
    return _hash_files(_ROOT, [*(_ROOT / "ensemble_snow").rglob("*.py"), *(_ROOT / "site").rglob("*")])


def fetch_fingerprint() -> str:
    """Hash of what decides the content of a *decoded download*: the fetchers, the GRIB reader, the regridder and the grid / level settings.
    The snow-ratio, statistics and consensus code are deliberately not in it, so changing them does not trigger a new download."""
    import hashlib
    pkg = _ROOT / "ensemble_snow"
    files = [*(pkg / "sources").glob("*.py"), *(pkg / n for n in ("grib.py", "regrid.py", "accum.py", "thermo.py", "http.py", "fields.py"))]
    h = hashlib.sha1(_hash_files(_ROOT, files).encode())
    for name in ("GRID_LON0", "GRID_LON1", "GRID_LAT0", "GRID_LAT1", "GRID_STEP", "CROP_PAD_DEG", "T_LEVELS", "RH_LEVELS", "Z_LEVELS", "EPS_PL_STEP_H"):
        h.update(f"{name}={getattr(config, name, None)!r}".encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- caches (kept between CI runs by actions/cache)
def _use_cache_for_reading() -> bool:
    return os.environ.get("ENSNOW_NO_CACHE", "").lower() not in ("1", "true", "yes")


def _decoded_path(cache: Path, key: str, cycle: datetime, steps: list[int], ffp: str) -> Path:
    import hashlib
    sig = hashlib.sha1(",".join(map(str, steps)).encode()).hexdigest()[:6]
    return cache / "decoded" / f"{key}_{cycle:%Y%m%d%H}_{sig}_{ffp}.pkl.gz"


def load_decoded(path: Path) -> EnsembleData | None:
    """A previously downloaded and decoded ensemble (on the display grid, ~100 MB for EPS instead of ~18 GB of GRIB), or None."""
    import gzip
    import pickle
    if not path.exists() or not _use_cache_for_reading():
        return None
    try:
        with gzip.open(path, "rb") as f:
            ens = pickle.load(f)
        return ens if isinstance(ens, EnsembleData) else None
    except Exception as exc:
        log.warning("could not read the cached download %s (%s); downloading again", path.name, exc)
        return None


def save_decoded(path: Path, ens: EnsembleData) -> None:
    import gzip
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(ens, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    for old in path.parent.glob(f"{ens.key}_*.pkl.gz"):             # keep only the newest download of each system: the cache stays small
        if old != path:
            old.unlink(missing_ok=True)
    log.info("cached the %s download (%.0f MB)", ens.key, path.stat().st_size / 1e6)


def _bundle_dir(cache: Path, pfp: str, cycles: dict[str, datetime], max_fhr: int) -> Path:
    sig = "-".join(f"{k}{t:%Y%m%d%H}" for k, t in sorted(cycles.items()))
    return cache / "bundles" / f"{pfp}_{sig}_f{max_fhr}"


def restore_bundles(src: Path, out: Path) -> dict | None:
    """Finished data files of an earlier build of the same pipeline code and model runs, copied into ``out/data``; returns their manifest."""
    mf = src / "manifest.json"
    if not mf.exists() or not _use_cache_for_reading():
        return None
    try:
        manifest = json.loads(mf.read_text())
        files = [e["file"] for ens in manifest["ensembles"].values() for e in ens["files"].values()]
        if not all((src / f).exists() for f in files):
            return None
        (out / "data").mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(src / f, out / "data" / f)
        return manifest
    except Exception as exc:
        log.warning("could not reuse the cached data files (%s); rebuilding", exc)
        return None


def save_bundles(dst: Path, out: Path, manifest: dict) -> None:
    if dst.parent.exists():
        shutil.rmtree(dst.parent, ignore_errors=True)                 # one set only
    dst.mkdir(parents=True, exist_ok=True)
    for ens in manifest["ensembles"].values():
        for e in ens["files"].values():
            shutil.copy2(out / "data" / e["file"], dst / e["file"])
    (dst / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


def already_deployed(url: str, cycles: dict[str, datetime], fingerprint: str, fetcher=None) -> bool:
    """True when the live site shows exactly these cycles built by this code (so nothing needs to be redone)."""
    import requests
    try:
        r = (fetcher or requests).get(url.rstrip("/") + "/data/manifest.json", timeout=30)
        if r.status_code != 200:
            return False
        live = r.json()
    except Exception as exc:                  # a missing or unreadable site just means: build
        log.info("could not read the deployed manifest (%s); building", exc)
        return False
    if live.get("fingerprint") != fingerprint or live.get("synthetic"):
        return False
    if any(live.get("ensembles", {}).get(k, {}).get("partial") for k in cycles):       # an incomplete download: a later build may do better
        return False
    want = {k: t.strftime("%Y-%m-%dT%H:%M:%SZ") for k, t in cycles.items()}
    have = {k: live.get("ensembles", {}).get(k, {}).get("init") for k in cycles}
    return want == have


def build_site(out: Path, keys: list[str], *, synthetic: bool, max_fhr: int, cache: Path, force_cycle: datetime | None, common: bool,
               workers: int, with_gis: bool = True, now: datetime | None = None, fetcher=None, skip_if_deployed: str | None = None,
               use_cache: bool | None = None, blend_weights: str | None = None) -> dict | None:
    blend_weights = parse_blend_weights(blend_weights if blend_weights is not None else os.environ.get("ENSNOW_BLEND_WEIGHTS"), keys)
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    lon, lat = config.grid_axes()
    if synthetic:
        from .sources.synthetic import SyntheticSource
        sources = {k: SyntheticSource(k) for k in keys}
        cycles = {k: force_cycle or now.replace(hour=(now.hour // 12) * 12, minute=0, second=0, microsecond=0) for k in keys}
    else:
        from .sources.eps import EpsSource
        from .sources.gefs import GefsSource
        from .sources.geps import GepsSource
        from .http import Fetcher
        f = fetcher or Fetcher(workers=max(workers, 8))
        classes = {"eps": EpsSource, "gefs": GefsSource, "geps": GepsSource}
        sources = {k: classes[k](f) for k in keys}
        cycles = pick_cycles(sources, now, max_fhr, common, force_cycle)
    if not cycles:
        raise SystemExit("no complete forecast run found for any requested system")

    wtag = "" if blend_weights == "equal" else "-w" + hashlib.sha1(json.dumps(blend_weights, sort_keys=True).encode()).hexdigest()[:6]   # the default keeps its old fingerprint
    fingerprint = code_fingerprint() + wtag
    if skip_if_deployed and not synthetic and already_deployed(skip_if_deployed, cycles, fingerprint):
        log.info("the live site already shows %s; nothing to do", {k: f"{t:%Y-%m-%d %H}Z" for k, t in cycles.items()})
        return None
    use_cache = (not synthetic) if use_cache is None else use_cache
    steps = forecast_steps(max_fhr, config.OUT_STEP_H, True, config.COARSE_AFTER_H, config.COARSE_STEP_H)
    out.mkdir(parents=True, exist_ok=True)

    # Same pipeline code and same model runs as an earlier build (only the page changed): reuse its finished data files, nothing to download or compute.
    pfp = pipeline_fingerprint() + wtag
    bdir = _bundle_dir(cache, pfp, cycles, max_fhr)
    if use_cache and set(cycles) == set(keys):
        manifest = restore_bundles(bdir, out)
        if manifest:
            log.info("reusing the data files built earlier from the same code and model runs: only the page is new")
            manifest["fingerprint"] = fingerprint
            return _finish_site(out, manifest, cache, with_gis, synthetic, fetcher)

    manifest_ens, kept = {}, {}
    ffp = fetch_fingerprint()
    for k in keys:
        if use_cache and hasattr(sources.get(k), "part_root"):             # a download that can be resumed keeps its finished hours here
            sources[k].part_root, sources[k].part_tag = cache, ffp
        if k not in cycles:
            continue
        log.info("=== %s  %s ===", k, cycles[k])
        try:
            dpath = _decoded_path(cache, k, cycles[k], steps, ffp)
            ens = load_decoded(dpath) if use_cache else None
            if ens is not None:
                log.info("%s: using the download cached from an earlier build (no data fetched)", k)
            else:
                ens = sources[k].load(cycles[k], steps, workers)
                if use_cache and not ens.partial:                     # an incomplete download is not worth reusing: the next attempt resumes it
                    try:
                        save_decoded(dpath, ens)
                    except Exception:
                        log.exception("%s: could not cache the download (the build goes on)", k)
            manifest_ens[k] = process_ensemble(ens, out, lat, lon, kept)
            if ens.partial:
                manifest_ens[k]["partial"] = True
        except Exception:
            log.exception("%s failed; continuing without it", k)
    if not manifest_ens:
        raise SystemExit("every system failed")
    comb = combine(kept, out, blend_weights)
    if comb:
        manifest_ens["all"] = comb

    manifest = {
        "version": 1, "fingerprint": fingerprint, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "synthetic": synthetic,
        "grid": {"lon0": float(lon[0]), "lat0": float(lat[0]), "dlon": config.GRID_STEP, "dlat": config.GRID_STEP, "nx": len(lon), "ny": len(lat)},
        "bounds": list(config.NEW_ENGLAND_BOUNDS), "stats": STAT_NAMES, "thresholds": list(config.THRESHOLDS_IN), "qpf_thresholds": list(config.QPF_THRESHOLDS_IN), "windows": list(config.WINDOWS_H),
        "out_step": config.OUT_STEP_H, "amt_step": config.AMT_STEP_IN, "slr_step": config.SLR_STEP,
        "methods": method_info(), "ensembles": manifest_ens,
    }
    if use_cache and set(keys) <= set(manifest_ens) and set(cycles) == set(keys) and not any(e.get("partial") for e in manifest_ens.values()):
        # a build missing a system, or with an incomplete download, is not worth reusing
        (out / "data").mkdir(parents=True, exist_ok=True)
        try:
            save_bundles(bdir, out, manifest)
        except Exception:
            log.exception("could not cache the finished data files (the build goes on)")
    return _finish_site(out, manifest, cache, with_gis, synthetic, fetcher)


def _finish_site(out: Path, manifest: dict, cache: Path, with_gis: bool, synthetic: bool = False, fetcher=None) -> dict:
    """Write the manifest, copy the page, build the terrain file and the GIS layers."""
    (out / "data").mkdir(parents=True, exist_ok=True)
    manifest.pop("terrain", None)
    terr = terrain.build(out / "data", cache, synthetic, fetcher) if with_gis else None
    if terr:
        manifest["terrain"] = terr
    (out / "data" / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    site = _ROOT / "site"
    if site.exists():
        shutil.copytree(site, out, dirs_exist_ok=True)
    if with_gis:
        gis.build_all(out / "data", cache / "ne")
    (out / ".nojekyll").write_text("")
    return manifest


def probe(keys: list[str], max_fhr: int, now: datetime | None = None) -> int:
    from .sources.eps import EpsSource
    from .sources.gefs import GefsSource
    from .sources.geps import GepsSource
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    classes = {"eps": EpsSource, "gefs": GefsSource, "geps": GepsSource}
    bad = 0
    for k in keys:
        src = classes[k]()
        for t in candidate_cycles(now, config.ENSEMBLES[k].cycles, 4):
            ok = src.is_complete(t, max_fhr)
            print(f"{k:5s} {t:%Y-%m-%d %H}Z  last step (F{max_fhr:03d}) {'AVAILABLE' if ok else 'missing'}")
            if k == "eps" and ok:
                print("      index params:", sorted(src.probe(t, max_fhr)) or "(could not read the index)")
            bad += (not ok)
    return 0 if bad < len(keys) * 4 else 1


def start_resource_monitor(interval: float = 30.0) -> None:
    """Log memory and disk use every ``interval`` seconds (Linux only), so a runner that is stopped mid-build leaves evidence in the log."""
    if not Path("/proc/meminfo").exists():
        return
    import threading
    import time

    def run():
        while True:
            try:
                mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo")}
                rss = next(int(ln.split()[1]) for ln in open("/proc/self/status") if ln.startswith("VmRSS"))
                free = shutil.disk_usage(".").free / 1e9
                log.info("resources: this process %d MB, memory available %d of %d MB, disk free %.1f GB",
                         rss // 1024, mem["MemAvailable"] // 1024, mem["MemTotal"] // 1024, free)
            except Exception:
                return
            time.sleep(interval)

    threading.Thread(target=run, daemon=True, name="resource-monitor").start()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--ensembles", default="eps,gefs,geps")
    ap.add_argument("--max-fhr", type=int, default=config.DEFAULT_MAX_FHR)
    ap.add_argument("--cycle", help="force this initialisation time, YYYYMMDDHH (UTC) for every system")
    ap.add_argument("--independent", action="store_true", help="use each system's own newest run instead of a common initialisation time")
    ap.add_argument("--synthetic", action="store_true", help="demo data (no network for the models)")
    ap.add_argument("--blend-weights", default=None, metavar="SPEC",
                    help='weights of the systems in the combined view: "equal" (default), "members" (by member count) or e.g. "eps=50,gefs=30,geps=20"')
    ap.add_argument("--no-gis", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="ignore the cached downloads and finished data files (they are still refreshed)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--probe", action="store_true", help="only report which forecast runs are on the servers")
    ap.add_argument("--skip-if-deployed", metavar="URL", help="exit without building when the site at URL already shows the newest runs from the same code")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    keys = [k.strip() for k in a.ensembles.split(",") if k.strip()]
    for k in keys:
        if k not in config.ENSEMBLES:
            ap.error(f"unknown ensemble {k!r}")
    try:
        parse_blend_weights(a.blend_weights if a.blend_weights is not None else os.environ.get("ENSNOW_BLEND_WEIGHTS"), keys)
    except ValueError as e:
        ap.error(str(e))
    if a.probe:
        return probe(keys, a.max_fhr)
    start_resource_monitor()
    if a.no_cache:
        os.environ["ENSNOW_NO_CACHE"] = "1"
    force = datetime.strptime(a.cycle, "%Y%m%d%H") if a.cycle else None
    build_site(Path(a.out), keys, synthetic=a.synthetic, max_fhr=a.max_fhr, cache=Path(a.cache), force_cycle=force, common=not a.independent,
               workers=a.workers, with_gis=not a.no_gis, skip_if_deployed=a.skip_if_deployed, blend_weights=a.blend_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
