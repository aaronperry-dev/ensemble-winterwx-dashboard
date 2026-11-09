"""Build the static site:  fetch -> snowfall by every method -> ensemble statistics -> compact files + manifest -> copy the page.

    python -m ensemble_snow.build --out public                    # newest complete run of every system
    python -m ensemble_snow.build --out public --synthetic        # demo data, no network (the page says so)
    python -m ensemble_snow.build --probe                         # check which runs / files are reachable, change nothing

No data are ever committed: GitHub Actions runs this, uploads ``public/`` as a Pages artifact and deploys it.
"""
from __future__ import annotations

import argparse
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

from . import compute, config, consensus, encode, gis, slr, stats
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
    if force is not None:
        return {k: force for k in sources}
    per = {k: candidate_cycles(now, config.ENSEMBLES[k].cycles) for k in sources}
    if common and len(sources) > 1:
        shared = sorted(set.intersection(*[set(v) for v in per.values()]), reverse=True)
        for t in shared[:6]:
            if all(src.is_complete(t, max_fhr) for src in sources.values()):
                return {k: t for k in sources}
        log.warning("no cycle is complete for every system; using each system's own newest run")
    out = {}
    for k, src in sources.items():
        for t in per[k]:
            if src.is_complete(t, max_fhr):
                out[k] = t
                break
        else:
            log.error("%s: no complete run found in the last %d cycles", k, len(per[k]))
    return out


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


def _bundle_job(key: str) -> tuple[str, int]:
    c = _CTX
    return key, write_bundle(c["dir"] / c["names"][key], c["cum"][key], c["liquid"], c["fhrs"], c["meta"][key], c["weights"], c["shifts"])


def write_bundles(out_dir: Path, names: dict[str, str], cum: dict[str, np.ndarray], liquid, fhrs, meta: dict[str, dict], weights, shifts) -> dict[str, int]:
    """One bundle per method, in parallel.  ``cum``: method -> (T, M, ny, nx).  Returns the size of each file."""
    done = _pmap(_bundle_job, list(names), dict(dir=out_dir, names=names, cum=cum, liquid=liquid, fhrs=fhrs, meta=meta, weights=weights, shifts=shifts))
    return dict(done)


def _blocks(cum: np.ndarray, liquid: np.ndarray, fhrs: list[int], weights: np.ndarray | None, shifts: dict[int, np.ndarray]):
    """cum, liquid: (T, M, ny, nx).  -> amt (W,S,T,ny,nx), prob (W,K,T,ny,nx), slr (T,ny,nx).  S = percentile statistics, then PMM / LPMM / SAM / SAM-LPM."""
    amt, prob = [], []
    T, _, ny, nx = cum.shape
    for w in config.WINDOWS_H:
        xw = stats.windowed(cum, w, fhrs)                                    # (T, M, ny, nx)
        a, p = stats.ensemble_stats(np.moveaxis(xw, 1, 0), weights)           # (S, T, ny, nx), (K, T, ny, nx)
        cons = np.empty((len(config.CONSENSUS_STATS), T, ny, nx), np.float32)
        for t in range(T):
            r = consensus.products(xw[t], shifts[w][t], weights)
            for i, k in enumerate(config.CONSENSUS_STATS):
                cons[i, t] = r[k]
        amt.append(np.concatenate([a, cons]))
        prob.append(p)
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


def write_bundle(path: Path, cum, liquid, fhrs, meta: dict, weights=None, shifts=None) -> int:
    if shifts is None:
        shifts = compute_shifts(cum, fhrs, weights)
    amt, prob, slr_mean = _blocks(cum, liquid, fhrs, weights, shifts)
    return encode.write_bundle(path, [
        dict(name="amt", data=amt, dtype="u16", lo=0.0, step=config.AMT_STEP_IN),
        dict(name="prob", data=prob, dtype="u8", lo=0.0, step=1.0),
        dict(name="slr", data=slr_mean, dtype="u8", lo=0.0, step=config.SLR_STEP),
    ], meta)


def bundle_meta(ensemble: str, method: str, init: str, fhrs: list[int]) -> dict:
    return {"ensemble": ensemble, "method": method, "init": init, "fhrs": fhrs, "stats": STAT_NAMES, "thresholds": list(config.THRESHOLDS_IN),
            "windows": list(config.WINDOWS_H)}


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
    names = {k: f"{ens.key}_{k}.bin.gz" for k in order}
    sizes = write_bundles(out / "data", names, res.methods, res.liquid_in, res.fhrs, {k: bundle_meta(ens.key, k, init, res.fhrs) for k in order}, None, shifts)
    for k in order:
        files[k] = {"file": names[k], "bytes": sizes[k]}
        log.info("%s/%s  %.0f kB", ens.key, k, sizes[k] / 1e3)
    if keep is not None:
        keep[ens.key] = (ens, res)
    notes = list(res.notes)
    return {"label": spec.label, "short": spec.short, "center": spec.center, "native": f"{spec.native}, {len(ens.members)} members", "license": spec.license, "url": spec.url,
            "init": ens.cycle.strftime("%Y-%m-%dT%H:%M:%SZ"), "members": len(ens.members), "fhrs": res.fhrs, "methods": order,
            "blend_members": res.blend_members, "ptype": res.ptype_source, "notes": notes, "files": files,
            "window_times": window_times(res.fhrs)}


def combine(kept: dict, out: Path) -> dict | None:
    """Pool systems that share an initialisation time into one ensemble in which every system carries equal total weight."""
    systems = [k for k in kept if True]
    if len(systems) < 2:
        return None
    inits = {kept[k][0].cycle for k in systems}
    if len(inits) != 1:
        log.info("combined view skipped: systems have different initialisation times")
        return None
    fhrs = sorted(set.intersection(*[set(kept[k][1].fhrs) for k in systems]))
    methods = [m for m in ["blend"] + list(slr.METHODS) if all(m in kept[k][1].methods for k in systems)]
    idx = {k: [kept[k][1].fhrs.index(f) for f in fhrs] for k in systems}
    weights = np.concatenate([np.full(len(kept[k][0].members), 1.0 / len(kept[k][0].members)) for k in systems])
    files = {}
    init = next(iter(inits)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pooled = {m: np.concatenate([kept[k][1].methods[m][idx[k]] for k in systems], axis=1) for m in methods}
    t0 = time.time()
    shifts = compute_shifts(pooled["blend"], fhrs, weights)
    log.info("all: spatial alignment of the members done in %.0f s", time.time() - t0)
    liq = np.concatenate([kept[k][1].liquid_in[idx[k]] for k in systems], axis=1)
    names = {m: f"all_{m}.bin.gz" for m in methods}
    sizes = write_bundles(out / "data", names, pooled, liq, fhrs, {m: bundle_meta("all", m, init, fhrs) for m in methods}, weights, shifts)
    for m in methods:
        files[m] = {"file": names[m], "bytes": sizes[m]}
    total = sum(len(kept[k][0].members) for k in systems)
    return {"label": "All systems combined", "short": "All", "center": "ECMWF + NOAA + ECCC",
            "native": f"{total} members; each system carries equal weight", "license": "see each system", "url": "",
            "init": next(iter(inits)).strftime("%Y-%m-%dT%H:%M:%SZ"), "members": total, "fhrs": fhrs, "methods": methods, "blend_members": [],
            "ptype": "per system", "notes": ["Systems are pooled with equal total weight (a GEPS member counts for more than an EPS member)."], "files": files,
            "window_times": window_times(fhrs)}


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


def pipeline_fingerprint() -> str:
    """Hash of the Python pipeline only.  The finished data files depend on nothing else (not on the page), so they are reused when it is unchanged."""
    return _hash_files(_ROOT, (_ROOT / "ensemble_snow").rglob("*.py"))


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
    want = {k: t.strftime("%Y-%m-%dT%H:%M:%SZ") for k, t in cycles.items()}
    have = {k: live.get("ensembles", {}).get(k, {}).get("init") for k in cycles}
    return want == have


def build_site(out: Path, keys: list[str], *, synthetic: bool, max_fhr: int, cache: Path, force_cycle: datetime | None, common: bool,
               workers: int, with_gis: bool = True, now: datetime | None = None, fetcher=None, skip_if_deployed: str | None = None,
               use_cache: bool | None = None) -> dict | None:
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

    fingerprint = code_fingerprint()
    if skip_if_deployed and not synthetic and already_deployed(skip_if_deployed, cycles, fingerprint):
        log.info("the live site already shows %s; nothing to do", {k: f"{t:%Y-%m-%d %H}Z" for k, t in cycles.items()})
        return None
    use_cache = (not synthetic) if use_cache is None else use_cache
    steps = forecast_steps(max_fhr, config.OUT_STEP_H, True, config.COARSE_AFTER_H, config.COARSE_STEP_H)
    out.mkdir(parents=True, exist_ok=True)

    # Same pipeline code and same model runs as an earlier build (only the page changed): reuse its finished data files, nothing to download or compute.
    pfp = pipeline_fingerprint()
    bdir = _bundle_dir(cache, pfp, cycles, max_fhr)
    if use_cache and set(cycles) == set(keys):
        manifest = restore_bundles(bdir, out)
        if manifest:
            log.info("reusing the data files built earlier from the same code and model runs: only the page is new")
            manifest["fingerprint"] = fingerprint
            return _finish_site(out, manifest, cache, with_gis)

    manifest_ens, kept = {}, {}
    ffp = fetch_fingerprint()
    for k in keys:
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
                if use_cache:
                    try:
                        save_decoded(dpath, ens)
                    except Exception:
                        log.exception("%s: could not cache the download (the build goes on)", k)
            manifest_ens[k] = process_ensemble(ens, out, lat, lon, kept)
        except Exception:
            log.exception("%s failed; continuing without it", k)
    if not manifest_ens:
        raise SystemExit("every system failed")
    comb = combine(kept, out)
    if comb:
        manifest_ens["all"] = comb

    manifest = {
        "version": 1, "fingerprint": fingerprint, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "synthetic": synthetic,
        "grid": {"lon0": float(lon[0]), "lat0": float(lat[0]), "dlon": config.GRID_STEP, "dlat": config.GRID_STEP, "nx": len(lon), "ny": len(lat)},
        "bounds": list(config.NEW_ENGLAND_BOUNDS), "stats": STAT_NAMES, "thresholds": list(config.THRESHOLDS_IN), "windows": list(config.WINDOWS_H),
        "out_step": config.OUT_STEP_H, "amt_step": config.AMT_STEP_IN, "slr_step": config.SLR_STEP,
        "methods": method_info(), "ensembles": manifest_ens,
    }
    if use_cache and set(keys) <= set(manifest_ens) and set(cycles) == set(keys):           # a build missing a system is not worth reusing
        (out / "data").mkdir(parents=True, exist_ok=True)
        try:
            save_bundles(bdir, out, manifest)
        except Exception:
            log.exception("could not cache the finished data files (the build goes on)")
    return _finish_site(out, manifest, cache, with_gis)


def _finish_site(out: Path, manifest: dict, cache: Path, with_gis: bool) -> dict:
    """Write the manifest, copy the page, build the GIS layers."""
    (out / "data").mkdir(parents=True, exist_ok=True)
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
    if a.probe:
        return probe(keys, a.max_fhr)
    start_resource_monitor()
    if a.no_cache:
        os.environ["ENSNOW_NO_CACHE"] = "1"
    force = datetime.strptime(a.cycle, "%Y%m%d%H") if a.cycle else None
    build_site(Path(a.out), keys, synthetic=a.synthetic, max_fhr=a.max_fhr, cache=Path(a.cache), force_cycle=force, common=not a.independent,
               workers=a.workers, with_gis=not a.no_gis, skip_if_deployed=a.skip_if_deployed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
