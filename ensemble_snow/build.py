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
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import compute, config, encode, gis, slr, stats
from .fields import EnsembleData
from .sources.base import forecast_steps

log = logging.getLogger("ensemble_snow")
STAT_NAMES = [f"p{p}" for p in config.PERCENTILES] + ["mean", "max"]
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
def _blocks(cum: np.ndarray, liquid: np.ndarray, fhrs: list[int], weights: np.ndarray | None):
    """cum, liquid: (T, M, ny, nx).  -> amt (W,S,T,ny,nx), prob (W,K,T,ny,nx), slr (T,ny,nx)."""
    amt, prob = [], []
    for w in config.WINDOWS_H:
        x = np.moveaxis(stats.windowed(cum, w, fhrs), 1, 0)                 # (M, T, ny, nx)
        a, p = stats.ensemble_stats(x, weights)
        amt.append(a)
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


def write_bundle(path: Path, cum, liquid, fhrs, meta: dict, weights=None) -> int:
    amt, prob, slr_mean = _blocks(cum, liquid, fhrs, weights)
    return encode.write_bundle(path, [
        dict(name="amt", data=amt, dtype="u16", lo=0.0, step=config.AMT_STEP_IN),
        dict(name="prob", data=prob, dtype="u8", lo=0.0, step=1.0),
        dict(name="slr", data=slr_mean, dtype="u8", lo=0.0, step=config.SLR_STEP),
    ], meta)


def method_info() -> dict:
    info = {"blend": {"label": BLEND_LABEL, "group": "blend", "desc": "Mean of the ratios of the blended methods (" + ", ".join(
        slr.METHODS[k].label for k in config.BLEND if k in slr.METHODS) + ") applied to the same snow-type liquid. Methods a system cannot compute are left out.",
        "ref": "", "in_blend": False}}
    for k, m in slr.METHODS.items():
        info[k] = {"label": m.label, "group": m.group, "desc": m.desc, "ref": m.ref, "in_blend": m.in_blend}
    return info


def process_ensemble(ens: EnsembleData, out: Path, grid_lat, grid_lon, keep: dict | None = None) -> dict:
    spec = config.ENSEMBLES[ens.key]
    LON, LAT = np.meshgrid(grid_lon, grid_lat)
    res = compute.compute_snowfall(ens, LAT, LON)
    (out / "data").mkdir(parents=True, exist_ok=True)
    files = {}
    order = ["blend"] + [k for k in slr.METHODS if k in res.methods]
    for k in order:
        meta = {"ensemble": ens.key, "method": k, "init": ens.cycle.strftime("%Y-%m-%dT%H:%M:%SZ"), "fhrs": res.fhrs,
                "stats": STAT_NAMES, "thresholds": list(config.THRESHOLDS_IN), "windows": list(config.WINDOWS_H)}
        name = f"{ens.key}_{k}.bin.gz"
        n = write_bundle(out / "data" / name, res.methods[k], res.liquid_in, res.fhrs, meta)
        files[k] = {"file": name, "bytes": n}
        log.info("%s/%s  %.0f kB", ens.key, k, n / 1e3)
    if keep is not None:
        keep[ens.key] = (ens, res)
    notes = list(res.notes)
    return {"label": spec.label, "short": spec.short, "center": spec.center, "native": f"{spec.native}, {len(ens.members)} members", "license": spec.license, "url": spec.url,
            "init": ens.cycle.strftime("%Y-%m-%dT%H:%M:%SZ"), "members": len(ens.members), "fhrs": res.fhrs, "methods": order,
            "blend_members": res.blend_members, "ptype": res.ptype_source, "notes": notes, "files": files}


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
    for m in methods:
        cum = np.concatenate([kept[k][1].methods[m][idx[k]] for k in systems], axis=1)
        liq = np.concatenate([kept[k][1].liquid_in[idx[k]] for k in systems], axis=1)
        meta = {"ensemble": "all", "method": m, "init": next(iter(inits)).strftime("%Y-%m-%dT%H:%M:%SZ"), "fhrs": fhrs, "stats": STAT_NAMES,
                "thresholds": list(config.THRESHOLDS_IN), "windows": list(config.WINDOWS_H)}
        n = write_bundle(out / "data" / f"all_{m}.bin.gz", cum, liq, fhrs, meta, weights)
        files[m] = {"file": f"all_{m}.bin.gz", "bytes": n}
    total = sum(len(kept[k][0].members) for k in systems)
    return {"label": "All systems combined", "short": "All", "center": "ECMWF + NOAA + ECCC",
            "native": f"{total} members; each system carries equal weight", "license": "see each system", "url": "",
            "init": next(iter(inits)).strftime("%Y-%m-%dT%H:%M:%SZ"), "members": total, "fhrs": fhrs, "methods": methods, "blend_members": [],
            "ptype": "per system", "notes": ["Systems are pooled with equal total weight (a GEPS member counts for more than an EPS member)."], "files": files}


# --------------------------------------------------------------------------- site
def code_fingerprint() -> str:
    """Hash of the pipeline and viewer sources: a code change rebuilds the site even when the model cycle has not changed."""
    import hashlib
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha1()
    for p in sorted([*(root / "ensemble_snow").rglob("*.py"), *(root / "site").rglob("*")]):
        if p.is_file() and "vendor" not in p.parts and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


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
               workers: int, with_gis: bool = True, now: datetime | None = None, fetcher=None, skip_if_deployed: str | None = None) -> dict | None:
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
    steps = forecast_steps(max_fhr, config.OUT_STEP_H)
    manifest_ens, kept = {}, {}
    out.mkdir(parents=True, exist_ok=True)
    for k in keys:
        if k not in cycles:
            continue
        log.info("=== %s  %s ===", k, cycles[k])
        try:
            ens = sources[k].load(cycles[k], steps, workers)
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
    (out / "data" / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

    site = Path(__file__).resolve().parent.parent / "site"
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
    force = datetime.strptime(a.cycle, "%Y%m%d%H") if a.cycle else None
    build_site(Path(a.out), keys, synthetic=a.synthetic, max_fhr=a.max_fhr, cache=Path(a.cache), force_cycle=force, common=not a.independent,
               workers=a.workers, with_gis=not a.no_gis, skip_if_deployed=a.skip_if_deployed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
